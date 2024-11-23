# What drives the price of a car?
Our goal is to understand what factors make a car more or less expensive. We are using a dataset from Kaggle that contains information on 426k used car sales. As a result of our analysis, we should provide clear recommendations to our client -- a used car dealership -- as to what consumers value in a used car. We will use the CRISP-DM framework for this project.

## Business Understanding
Rephrasing this into a technical data science problem, can we accurately predict the price of a used car given different features of said car? Which features are the most important for predicting price? After determining the most important features, predict the price of a car using a regression model.

## Data Understanding
To get familiar with the data, I will first examine the different rows and columns in the dataset. What features do we have? We can start to think about which features might be useful and which might not be. We can think back to our business problem to determine which features the customer actually cares about. We might be able to use some common sense and remove any columns that we know for sure will not be useful to us.

We will also use graphs to visualize the data and get a better understanding of it. Visualizing the data may help us identify if we need to perform any transformations.

After initial examination of the data, I saw some data that I wanted to remove. There were ID and VIN columns that will not help us solve our problem or help predict price. I also wanted to remove the locations (state and region) columns of our dataframe because I believe the location of the sale might not be very important to our customer. I saw a lot of missing values in this data that needed to be cleaned up as well.

I saw there were some outliers in our data that are most likely input errors that need to be removed. The max value of the price column is over $3 billion and the max value for the odometer is 10 million. Diving deeper into the price and odometer columns, I saw that there were a lot of absurdly large values in both columns. I also saw a lot of cars that were the price of 0 or close to it. These might've been gifts and I believe this will not be of any use to our customer.

Then I looked into the car models and saw that there might be a lot of different entries that all relate to the same car. I saw "silverado 1500", "1500", and "silverado". I think there was just too many vehicle models here for it to be useful for our customer.

## Data Preparation
Here I began cleaning the data, applying transformations, and creating the final dataset to be used for modeling. I first dropped the ID, VIN, Region, State, and model columns. I do not believe these would've been useful to the customer or help us predict price. In order to deal with outliers in the price and odometer columns, I used the IQR method to remove those data points higher or lower than 1.5 * IQR. I am only interested in cars that sold for more than $1000 because the customer is trying to make money and these $0 priced "sales" won't matter to them. Generally the most amount of miles that a car can be expected to make it to is 300k so I was happy that the resulting largest odometer values were around 277k.

Without knowing more information about our customer, I'm going to assume that as a used car salesman he is probably not interested in the sale of very expensive luxury or sports cars. Removing the high price datapoints should remove these cars from our dataset.

I then plotted boxplots of all our categorical columns:
![image](https://github.com/user-attachments/assets/613198b1-85d5-4def-97b8-54209f43fab5)

![image](https://github.com/user-attachments/assets/bc4ca6a0-1151-420c-b0f0-aea2945f79f7)

![image](https://github.com/user-attachments/assets/f6cd21ca-5488-43f7-875c-76193d743eef)

![image](https://github.com/user-attachments/assets/e401cf3b-389c-4ef5-9fad-2941973d8202)

![image](https://github.com/user-attachments/assets/126c8f06-e64c-4434-b92f-f7408d31e48f)

![image](https://github.com/user-attachments/assets/c35d47d0-71fd-4b20-86e0-2f3451cd79eb)

![image](https://github.com/user-attachments/assets/22ef75fb-6a9a-4bc4-897f-c45666b87c46)

![image](https://github.com/user-attachments/assets/c0edc442-e1f3-4eaf-9fdf-02a21d02d1da)

![image](https://github.com/user-attachments/assets/764c6d72-5ecc-41d2-a1d6-0f2051b1fb29)

![image](https://github.com/user-attachments/assets/75df4ddf-7a61-47bc-a851-d3ce77aa91eb)

After looking at these plots and in an effort to further reduce dimensionality/overfitting, I removed the manufacturer and color columns from our dataset. Without knowing more information about the customer, I'm going to assume they are more interested in other aspects of the cars.

Features like condition of the car, title status, type, and mileage might be of more value.

I then plotted the numerical columns of our dataset and created a heatmap:

![image](https://github.com/user-attachments/assets/3a94c4ce-b781-4fe0-9bd0-ceb6b9f2e22f)

![image](https://github.com/user-attachments/assets/a489066d-e5ee-4075-b554-0a699fd73913)

![image](https://github.com/user-attachments/assets/73e1bb56-b2f3-4fbd-9391-8a0d36e03d4d)

With this heatmap we can see that price and odometer are negatively correlated, which is expected. The year does not seem to be closely correlated with price.

I converted all of our categorical data into dummy variables to prepare for our modeling. I created another correlation matrix for the price column. None of these values were highly correlated with price. I chose these columns with a correlation of >= 0.10 or <= -0.10 to determine which columns of our original dataset to keep.

## Modeling
We will be building models with three different regression algorithms: Linear, Ridge, and Lasso. We will use Recursive Feature Elimination to determine the best number of features to use for each.

First, we separate our training data from our target 'price' column. Then we perform a train/test split. I created 3 pipelines to use for our models and used GridSearchCV to find the best hyperparameters for our models. I decided to have it try between 1 and 2 polynomial degree and between 4, 8, or 12 features. For the ridge and lasso algorithms, we searched between alpha values of 0.01, 0.1, 1, 10 and 100 for the best alpha hyperparameter.

## Evaluation

## Deployment





