import numpy as np
import pandas as pd
import matplotlib
import io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from flask import Flask, render_template, send_file, request
from matplotlib.ticker import FuncFormatter

# Sets rendering in matplot for plots to files rather than displaying them
matplotlib.use("Agg")

# Initialize Flask application
app = Flask(__name__)


# Converts a date string (MM-DD) to the corresponding day number of the year
def date_to_day_number(date_str):
    if pd.isnull(date_str):
        return np.nan
    # Parses the date string to a datetime object
    date = datetime.strptime(date_str, "%m-%d")
    # Creates a datetime object for the first day of the same year
    new_year_day = datetime(date.year, 1, 1)
    # Calculates the day number as the difference in days plus one (to start counting from 1)
    day_number = (date - new_year_day).days + 1
    return day_number


# Converts a day number back to a date string (MM-DD)
def day_number_to_date_str(day_number):
    # Creates a datetime object for the first day of the current year
    date = datetime(datetime.now().year, 1, 1) + timedelta(days=day_number - 1)
    return date.strftime("%m-%d")


# plots regression line and predictions for a specific site
def plot_regression(site_name, data, spread):
    site_data = data[data["Site Name"] == site_name]

    # Drop rows with no values
    site_data = site_data.dropna(subset=["Day"])

    # Identify and remove the most recent data point
    most_recent_year = site_data["Year"].max()
    most_recent_data_point = site_data[site_data["Year"] == most_recent_year]
    site_data = site_data[site_data["Year"] != most_recent_year]

    # Reshape the Year column to a 2D array for sklearn and get day column values as target variables
    x = site_data["Year"].values.reshape(-1, 1)
    y = site_data["Day"].values

    # Initialize the linear regression model
    model = LinearRegression()
    model.fit(x, y)

    # Predict the day values based on the model (used for the regression line later)
    y_pred = model.predict(x)

    # Predict the bloom day for the next year
    next_year = np.array([[x.max() + 1]])  # Increments the maximum year by 1
    next_year_pred = model.predict(next_year)[0]

    # Round the predicted values to the nearest integer
    next_year_pred_rounded = round(next_year_pred)

    # Evaluate the model using the most recent data point
    most_recent_year_value = most_recent_data_point["Year"].values[0]
    most_recent_day_value = most_recent_data_point["Day"].values[0]
    predicted_value = model.predict(np.array([[most_recent_year_value]]))[0]
    predicted_value_rounded = round(predicted_value)
    percentage_error = abs(predicted_value - most_recent_day_value) / most_recent_day_value * 100
    percentage_error_rounded = round(percentage_error, 2)

    print(f"City: {site_name}")
    print(f"Predicted value: {predicted_value_rounded} or {day_number_to_date_str(predicted_value_rounded)}")
    print(f"Actual value: {most_recent_day_value} or {day_number_to_date_str(most_recent_day_value)}")
    print(f"Percentage error: {percentage_error_rounded}%")

    # Generates the spread (green area on graphs), spead is set to 7 but can be changed to fit a bigger model
    lower_bound = next_year_pred - spread
    upper_bound = next_year_pred + spread

    # Calculates the percentage of observed bloom dates within the spread range
    within_spread = np.sum((y >= lower_bound) & (y <= upper_bound))
    total_observations = len(y)
    percentage_within_spread = (within_spread / total_observations) * 100

    # Plots the observed data, regression lines, and spread
    fig, axes = plt.subplots(figsize=(20, 6))
    fig.subplots_adjust(right=0.5)  # Adjust the right margin to make space for the annotation
    axes.scatter(x, y, color="blue", label="Observed")
    axes.plot(x, y_pred, color="red", label="Regression Line")
    # Plots the end of the regression line for the next year with a horizontal line
    axes.axhline(next_year_pred_rounded, color="green", linestyle="--",
               label=f"Next Year Prediction ({day_number_to_date_str(int(next_year_pred_rounded))})")
    # Fills the area between the lower and upper bounds to indicate prediction spread
    axes.fill_between([x.min(), x.max() + 1], lower_bound, upper_bound, color="green", alpha=0.1,
                    label=f"Spread ({day_number_to_date_str(int(lower_bound))} - "
                          f"{day_number_to_date_str(int(upper_bound))})")
    # Displays the percentage chance of bloom within the spread
    axes.annotate(f"{percentage_within_spread:.2f}% chance of seeing bloom",
                xy=(1.02, 0.5), xycoords="axes fraction",
                fontsize=12, verticalalignment="center", bbox=dict(facecolor="white", alpha=0.8))

    # Keys for the graph
    axes.set_title(f"Regression Line for {site_name}")
    axes.set_xlabel("Year")
    axes.set_ylabel("Month and Day")

    # Formatter for the y-axis to show month/day instead of day numbers
    axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: day_number_to_date_str(int(y))))

    # Displays the legend and adds grid lines
    axes.legend()
    axes.grid(True)

    # Uses BytesIO to save images in memory rather than saving to a file
    img = io.BytesIO()  # Initializes the object
    fig.savefig(img, format="png")  # Saves the figure generated to the object as a PNG
    img.seek(0)  # Moves the file pointer to the beginning of the object
    plt.close(fig)  # Close the figure
    return img  # Returns the object containing the image data


# Function that creates the 5-year average display
def plot_5_year_average(data):
    # Groups the data by 5-year intervals
    data["5YearPeriod"] = (data["Year"] // 5) * 5
    avg_data = data.groupby("5YearPeriod")["Day"].mean().reset_index()

    # Plots the 5-year average
    fig, axes = plt.subplots(figsize=(14, 6))
    axes.plot(avg_data["5YearPeriod"], avg_data["Day"], marker="o", linestyle="-", color="black")
    axes.set_title("5-year Average of Cherry Blossom's First-bloom Date Across All of Japan")
    axes.set_xlabel("Year")
    axes.set_ylabel("Month and Day")
    axes.grid(True)

    # Formatter for the y-axis to show month/day instead of day numbers
    axes.yaxis.set_major_formatter(FuncFormatter(lambda y, _: day_number_to_date_str(int(y))))

    # Uses BytesIO to save image to memory rather than saving to a file
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    plt.close(fig)
    return img


# Function that creates the bar graph
def plot_bloom_by_date_periods(data):
    # Filters data for the last 10 years
    last_10_years = data[data["Year"] >= data["Year"].max() - 9]

    # Calculates average bloom dates for each city
    avg_bloom_dates = last_10_years.groupby("Site Name")["Day"].mean().reset_index()

    # Defines one-week periods, capping at the beginning of June(day number 160 for June 9th)
    periods = [(i, i + 6) for i in range(1, 160, 7)]

    # Group cities into one-week periods
    period_labels = []
    for start, end in periods:
        period_label = f"{day_number_to_date_str(start)} - {day_number_to_date_str(end)}"
        period_labels.append(period_label)

    avg_bloom_dates["Period"] = pd.cut(avg_bloom_dates["Day"], bins=[start for start, end in periods] + [160], labels=period_labels, right=False)

    # Plots the bar graph
    fig, axes = plt.subplots(figsize=(14, 8))
    period_counts = avg_bloom_dates["Period"].value_counts().sort_index()
    axes.barh(period_counts.index, period_counts.values, color="pink")
    axes.set_title("Number of Cities by Average Bloom Date")
    axes.set_xlabel("Number of Cities")
    axes.set_ylabel("One-Week Periods")
    axes.grid(True)

    # Uses BytesIO to save image to memory rather than saving to a file
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    plt.close(fig)
    return img


# Function to generate a graph of the percentage error for cities over a certain percentage.
# Can be modified to allow a range for finer troubleshooting
def plot_percentage_error(data):
    print("Generating percentage error plot")
    sites = data["Site Name"].unique()
    percentage_errors = []
    valid_sites = []

    for site in sites:
        site_data = data[data["Site Name"] == site]
        site_data = site_data.dropna(subset=["Day"])
        most_recent_year = site_data["Year"].max()
        validation_data = site_data[site_data["Year"] == most_recent_year]
        site_data = site_data[site_data["Year"] != most_recent_year]

        if len(validation_data) == 0 or len(site_data) == 0:
            continue

        x = site_data["Year"].values.reshape(-1, 1)
        y = site_data["Day"].values

        model = LinearRegression()
        model.fit(x, y)

        validation_year_value = validation_data["Year"].values[0]
        validation_day_value = validation_data["Day"].values[0]
        predicted_value = model.predict(np.array([[validation_year_value]]))[0]
        percentage_error = abs(predicted_value - validation_day_value) / validation_day_value * 100

        if percentage_error > 25:
            percentage_errors.append(percentage_error)
            valid_sites.append(site)

    # Plots the graph
    fig, axes = plt.subplots(figsize=(14, 8))
    axes.bar(valid_sites, percentage_errors, color="pink")
    axes.set_title("Percentage Error for Predicted Bloom Dates (Above 25%)")
    axes.set_xlabel("Cities")
    axes.set_ylabel("Percentage Error")
    axes.set_xticklabels(valid_sites, rotation=90)
    axes.grid(True)

    # Uses BytesIO to save image to memory rather than saving to a file
    img = io.BytesIO()
    fig.savefig(img, format="png")
    img.seek(0)
    plt.close(fig)
    return img


# Read the CSV file containing Sakura bloom dates
file_path = "Data/sakura_first_bloom_dates.csv"
sakura_blossom_data = pd.read_csv(file_path)

# Converts all year columns to datetime, ignores the non-date columns
date_columns = sakura_blossom_data.columns[2:-1]
for col in date_columns:
    sakura_blossom_data[col] = pd.to_datetime(sakura_blossom_data[col])

# Removes the Currently Being Observed and 30-Year Average columns, adds NaN to empty values in the CSV
sanitized_blossoms = sakura_blossom_data.drop(columns=["Currently Being Observed", "30 Year Average 1981-2010"])
sanitized_blossoms = sanitized_blossoms.fillna(np.nan)

# Converts data to long format
melted_data = sanitized_blossoms.melt(id_vars=["Site Name"], var_name="Year", value_name="Day")
melted_data["Year"] = melted_data["Year"].astype(int)
melted_data["Day"] = pd.to_datetime(melted_data["Day"])

# Sorting the data so it's easier to manipulate later, converts the day column value to day numbers
melted_data["Day"] = melted_data["Day"].dt.strftime("%m-%d")
melted_data["Day"] = melted_data["Day"].apply(date_to_day_number)

# Lists the unique sites
unique_sites = melted_data["Site Name"].unique()

# Display the original data
# print("Original Data:")
# print(sakura_blossom_data.head(10))
# Display the cleansed/sanitized data
# print("\nCleansed/Sanitized Data:")
# print(sanitized_blossoms.head(10))
# Display the melted data
# print("\nMelted Data:")
# print(melted_data.head(10))


# For the index route to render the homepage
@app.route("/")
def index():
    # Get the first site name from the list of unique sites
    first_site = unique_sites[0]
    return render_template("index.html", sites=unique_sites, selected_site=first_site)


# For the plot route to generate and display the plot for a specific site
@app.route("/plot_image/<site_name>")
def plot_image(site_name):
    spread = int(request.args.get("spread", 7))  # Get the spread value from the query parameters, default to 7
    img = plot_regression(site_name, melted_data, spread)
    return send_file(img, mimetype="image/png")


# For the plot route to generate and display the 5 year average
@app.route("/plot_5_year_average")
def plot_5year_average_route():
    img = plot_5_year_average(melted_data)
    return send_file(img, mimetype="image/png")


# For the plot route to generate and display the bar graph
@app.route("/plot_bloom_by_date_periods")
def plot_bloom_by_date_periods_route():
    img = plot_bloom_by_date_periods(melted_data)
    return send_file(img, mimetype="image/png")


# For the plot route to generate and display the percentage error graph
@app.route("/plot_percentage_error")
def plot_percentage_error_route():
    img = plot_percentage_error(melted_data)
    return send_file(img, mimetype="image/png")


# Runs the Flask app
if __name__ == "__main__":
    app.run(debug=True)



