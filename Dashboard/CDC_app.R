library(shiny)
library(tidyverse)
library(caret)
library(pROC)

# load and prep data
raw <- read_csv("CDC-2019-2021-2023-DATA.csv", show_col_types = FALSE)

raw <- raw %>% filter(IYEAR != 2024)

# variables used in log reg and knn
used_vars <- c(
  "ADDEPEV3",
  "BIRTHSEX", "MENTHLTH", "POORHLTH",
  "DECIDE", "DIFFALON", "IYEAR",
  "ACEDEPRS", "ACEDRINK", "ACEDRUGS", "ACEPRISN",
  "ACEDIVRC", "ACEPUNCH", "ACEHURT1", "ACESWEAR",
  "ACETOUCH", "ACETTHEM", "ACEHVSEX"
)

df <- raw %>%
  select(all_of(used_vars)) %>%
  # keep only No / Yes responses
  filter(ADDEPEV3 %in% c("No", "Yes")) %>%
  mutate(
    ADDEPEV3 = factor(ADDEPEV3, levels = c("No", "Yes")),
    # categorical
    across(
      c(
        BIRTHSEX, DECIDE, DIFFALON,
        ACEDEPRS, ACEDRINK, ACEDRUGS, ACEPRISN,
        ACEDIVRC, ACEPUNCH, ACEHURT1, ACESWEAR,
        ACETOUCH, ACETTHEM, ACEHVSEX, IYEAR
      ),
      ~ factor(.x)
    )
  ) %>%
  drop_na()

set.seed(42)
if (nrow(df) > 50000) {
  df <- df %>% slice_sample(n = 50000)
}

#clear NAs 
stopifnot(sum(is.na(df)) == 0)

## ------------------  TRAIN / TEST SPLIT  ------------------ ##

set.seed(42)
train_index <- createDataPartition(df$ADDEPEV3, p = 0.8, list = FALSE)
train_data  <- df[train_index, ]
test_data   <- df[-train_index, ]

train_data <- na.omit(train_data)
test_data  <- na.omit(test_data)

stopifnot(sum(is.na(train_data)) == 0)
stopifnot(sum(is.na(test_data)) == 0)

# caret control

ctrl <- trainControl(
  method = "none",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# logist regression fitting

cat("Fitting logistic regression...\n")
logit_fit <- train(
  ADDEPEV3 ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = ctrl
)
cat("Logistic regression done.\n")

# log loss function

log_loss <- function(y_true, p_hat) {
  eps <- 1e-15
  p_hat <- pmin(pmax(p_hat, eps), 1 - eps)
  y_numeric <- ifelse(y_true == "Yes", 1, 0)
  -mean(y_numeric * log(p_hat) + (1 - y_numeric) * log(1 - p_hat))
}

## ------------------  SHINY UI  ------------------ ##

ui <- fluidPage(
  titlePanel("BRFSS Mental Health Prediction Dashboard"),
  
  sidebarLayout(
    sidebarPanel(
      h4("Model controls"),
      selectInput(
        "model_type",
        "Choose model:",
        choices = c("Logistic Regression" = "logit",
                    "k-Nearest Neighbors" = "knn")
      ),
      sliderInput(
        "threshold",
        "Classification threshold on P(depressed = Yes):",
        min = 0.1, max = 0.9, value = 0.5, step = 0.01
      ),
      conditionalPanel(
        condition = "input.model_type == 'knn'",
        sliderInput(
          "k",
          "k for k-NN:",
          min = 3, max = 51, value = 15, step = 2
        )
      ),
      
      hr(),
      h4("Single-person prediction"),
      helpText("Set predictor values for one individual:"),
      
      selectInput("BIRTHSEX", "Birth sex:",
                  choices = levels(train_data$BIRTHSEX)),
      sliderInput("MENTHLTH", "MENTHLTH (days of poor mental health):",
                  min = floor(min(train_data$MENTHLTH)),
                  max = ceiling(max(train_data$MENTHLTH)),
                  value = median(train_data$MENTHLTH)),
      sliderInput("POORHLTH", "days of poor physical health:",
                  min = floor(min(train_data$POORHLTH)),
                  max = ceiling(max(train_data$POORHLTH)),
                  value = median(train_data$POORHLTH)),
      selectInput("DECIDE", "DECIDE:",
                  choices = levels(train_data$DECIDE)),
      selectInput("DIFFALON", "DIFFALON:",
                  choices = levels(train_data$DIFFALON)),
      selectInput("IYEAR", "Survey year:",
                  choices = levels(train_data$IYEAR)),
      selectInput("ACEDEPRS", "ACEDEPRS:",
                  choices = levels(train_data$ACEDEPRS)),
      selectInput("ACEDRINK", "ACEDRINK:",
                  choices = levels(train_data$ACEDRINK)),
      selectInput("ACEDRUGS", "ACEDRUGS:",
                  choices = levels(train_data$ACEDRUGS)),
      selectInput("ACEPRISN", "ACEPRISN:",
                  choices = levels(train_data$ACEPRISN)),
      selectInput("ACEDIVRC", "ACEDIVRC:",
                  choices = levels(train_data$ACEDIVRC)),
      selectInput("ACEPUNCH", "ACEPUNCH:",
                  choices = levels(train_data$ACEPUNCH)),
      selectInput("ACEHURT1", "ACEHURT1:",
                  choices = levels(train_data$ACEHURT1)),
      selectInput("ACESWEAR", "ACESWEAR:",
                  choices = levels(train_data$ACESWEAR)),
      selectInput("ACETOUCH", "ACETOUCH:",
                  choices = levels(train_data$ACETOUCH)),
      selectInput("ACETTHEM", "ACETTHEM:",
                  choices = levels(train_data$ACETTHEM)),
      selectInput("ACEHVSEX", "ACEHVSEX:",
                  choices = levels(train_data$ACEHVSEX))
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Model performance",
                 br(),
                 h4("Metrics on test set"),
                 tableOutput("metrics_table"),
                 h4("Confusion matrix"),
                 verbatimTextOutput("cm_text")
        ),
        tabPanel("ROC curve",
                 br(),
                 plotOutput("roc_plot")
        ),
        tabPanel("Single prediction",
                 br(),
                 h4("Predicted probability of depression (\"Yes\")"),
                 verbatimTextOutput("single_prob"),
                 h4("Predicted label at current threshold"),
                 verbatimTextOutput("single_label")
        )
      )
    )
  )
)

## ------------------  SHINY SERVER  ------------------ ##

server <- function(input, output, session) {
  
  # k-NN model fit is reactive to allow changing k
  knn_fit <- reactive({
    req(input$k)
    train(
      ADDEPEV3 ~ .,
      data = train_data,
      method = "knn",
      tuneGrid = data.frame(k = input$k),
      metric = "ROC",
      trControl = ctrl,
      preProcess = c("center", "scale")
    )
  })
  
  # selecting which model
  current_model <- reactive({
    if (input$model_type == "logit") {
      logit_fit
    } else {
      knn_fit()
    }
  })
  
  # predicting prob 
  test_probs <- reactive({
    predict(current_model(), newdata = test_data, type = "prob")[, "Yes"]
  })
  
  # predicted labels using current threshold
  test_pred_labels <- reactive({
    factor(
      ifelse(test_probs() >= input$threshold, "Yes", "No"),
      levels = c("No", "Yes")
    )
  })
  
 # confustion matrix 
  output$cm_text <- renderPrint({
    cm <- confusionMatrix(
      data = test_pred_labels(),
      reference = test_data$ADDEPEV3,
      positive = "Yes"
    )
    cm
  })
  
  # metrics table
  output$metrics_table <- renderTable({
    probs <- test_probs()
    preds <- test_pred_labels()
    
    cm <- confusionMatrix(
      data = preds,
      reference = test_data$ADDEPEV3,
      positive = "Yes"
    )
    
    roc_obj <- roc(response = test_data$ADDEPEV3,
                   predictor = probs,
                   levels = c("No", "Yes"),
                   quiet = TRUE)
    
    data.frame(
      Accuracy    = round(cm$overall["Accuracy"], 3),
      Sensitivity = round(cm$byClass["Sensitivity"], 3),
      Specificity = round(cm$byClass["Specificity"], 3),
      AUC         = round(as.numeric(auc(roc_obj)), 3),
      LogLoss     = round(log_loss(test_data$ADDEPEV3, probs), 3)
    )
  })
  
  # ROC curve plot
  output$roc_plot <- renderPlot({
    probs <- test_probs()
    roc_obj <- roc(response = test_data$ADDEPEV3,
                   predictor = probs,
                   levels = c("No", "Yes"),
                   quiet = TRUE)
    
    plot(
      roc_obj,
      main = paste0(
        "ROC curve - ",
        ifelse(input$model_type == "logit",
               "Logistic Regression",
               paste0("k-NN (k = ", input$k, ")"))
      )
    )
    abline(a = 0, b = 1, lty = 2)
  })
  
  # Single-person observation assembled from inputs
  new_obs <- reactive({
    tibble(
      BIRTHSEX = factor(input$BIRTHSEX, levels = levels(train_data$BIRTHSEX)),
      MENTHLTH = as.numeric(input$MENTHLTH),
      POORHLTH = as.numeric(input$POORHLTH),
      DECIDE   = factor(input$DECIDE,   levels = levels(train_data$DECIDE)),
      DIFFALON = factor(input$DIFFALON, levels = levels(train_data$DIFFALON)),
      IYEAR    = factor(input$IYEAR,    levels = levels(train_data$IYEAR)),
      ACEDEPRS = factor(input$ACEDEPRS, levels = levels(train_data$ACEDEPRS)),
      ACEDRINK = factor(input$ACEDRINK, levels = levels(train_data$ACEDRINK)),
      ACEDRUGS = factor(input$ACEDRUGS, levels = levels(train_data$ACEDRUGS)),
      ACEPRISN = factor(input$ACEPRISN, levels = levels(train_data$ACEPRISN)),
      ACEDIVRC = factor(input$ACEDIVRC, levels = levels(train_data$ACEDIVRC)),
      ACEPUNCH = factor(input$ACEPUNCH, levels = levels(train_data$ACEPUNCH)),
      ACEHURT1 = factor(input$ACEHURT1, levels = levels(train_data$ACEHURT1)),
      ACESWEAR = factor(input$ACESWEAR, levels = levels(train_data$ACESWEAR)),
      ACETOUCH = factor(input$ACETOUCH, levels = levels(train_data$ACETOUCH)),
      ACETTHEM = factor(input$ACETTHEM, levels = levels(train_data$ACETTHEM)),
      ACEHVSEX = factor(input$ACEHVSEX, levels = levels(train_data$ACEHVSEX))
    )
  })
  
  # Single prediction probability
  output$single_prob <- renderPrint({
    probs <- predict(current_model(), newdata = new_obs(), type = "prob")[, "Yes"]
    round(probs, 3)
  })
  
  # Single prediction label
  output$single_label <- renderPrint({
    probs <- predict(current_model(), newdata = new_obs(), type = "prob")[, "Yes"]
    label <- ifelse(probs >= input$threshold, "Yes", "No")
    label
  })
}

shinyApp(ui, server)
