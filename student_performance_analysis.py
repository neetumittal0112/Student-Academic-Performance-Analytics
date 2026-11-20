# %% [markdown]
# # Student Academic Performance Analytics
# 
# ## Project Overview
# 
# This project analyses student academic performance using Python.
# 
# ### Libraries Used
# -Pandas
# -NumPy
# -Matplotlib

# %%
import pandas as pd

# %%
import numpy as np

# %%
import matplotlib.pyplot as plt

# %%
# Set random seed
np.random.seed(42)

# Number of students
n = 50

# Create student dataset
data = {
    "Student_ID": [f"S{i}" for i in range(101, 101 + n)],
    "Department": np.random.choice(["CSE", "IT", "ECE", "MECH"], n),
    "Attendance": np.random.randint(40, 101, n),
    "Study_Hours": np.random.randint(1, 9, n),
    "Python_Marks": np.random.randint(30, 101, n),
    "Maths_Marks": np.random.randint(25, 101, n),
    "English_Marks": np.random.randint(35, 101, n),
    "Internet_Access": np.random.choice(["Yes", "No"], n)
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Add missing values
df.loc[2, "Attendance"] = np.nan
df.loc[8, "Attendance"] = np.nan

# Add duplicate Student_ID
df.loc[49, "Student_ID"] = df.loc[0, "Student_ID"]

# Add invalid marks
df.loc[5, "Python_Marks"] = 105
df.loc[10, "Maths_Marks"] = -5

# Save dataset
df.to_csv("02_student_performance.csv", index=False)

print("Dataset created successfully!")

# %%
# Question 19: Load and inspect dataset

df = pd.read_csv("02_student_performance.csv")

print("First five rows:")
display(df.head())

print("Dataset shape:")
print(df.shape)

print("Dataset information:")
df.info()

print("Statistical summary:")
display(df.describe(include="all"))


# %%
# Question 20: Missing values and duplicate IDs

print("Missing values:")
print(df.isnull().sum())

print("Duplicate Student IDs:")
print(df["Student_ID"].duplicated().sum())

print("Duplicate rows:")
display(df[df["Student_ID"].duplicated(keep=False)])

# %%
# Question 21: Handle missing attendance values

median_attendance = df["Attendance"].median()

df["Attendance"] = df["Attendance"].fillna(median_attendance)

print("Median attendance:", median_attendance)
print("Remaining missing attendance:")
print(df["Attendance"].isnull().sum())

# %%
# Question 22: Check invalid marks

score_columns = [
    "Python_Marks",
    "Maths_Marks",
    "English_Marks"
]

for col in score_columns:
    invalid_rows = df[(df[col] < 0) | (df[col] > 100)]

    print("Invalid values in", col)
    display(invalid_rows[["Student_ID", col]])

    df.loc[(df[col] < 0) | (df[col] > 100), col] = np.nan

print("Invalid marks replaced with missing values.")

# %%
# Question 23: Calculate total and average marks

df["Total_Marks"] = df[score_columns].sum(axis=1, min_count=1)

df["Average_Marks"] = df[score_columns].mean(axis=1)

display(df[["Student_ID", "Total_Marks", "Average_Marks"]].head(10))


# %%
# Question 24: Create Grade column

df["Grade"] = pd.cut(
    df["Average_Marks"],
    bins=[0, 40, 60, 70, 80, 101],
    labels=["F", "D", "C", "B", "A"],
    right=False
)

print("Grade distribution:")
display(df["Grade"].value_counts().sort_index())

# %%
# Question 25: Department-wise average marks

department_avg = df.groupby("Department")["Average_Marks"].mean()

print("Department-wise average marks:")
display(department_avg)


# %%
# Question 26: Top 10 students

top_10 = df.nlargest(10, "Average_Marks")

print("Top 10 students:")
display(top_10[["Student_ID", "Department", "Average_Marks", "Grade"]])


# %%
# Question 27: Students below 40% average marks

at_risk = df[df["Average_Marks"] < 40]

print("Students needing academic support:")
display(at_risk[["Student_ID", "Department", "Average_Marks"]])

# %%
# Question 28: Attendance band analysis

df["Attendance_Band"] = pd.cut(
    df["Attendance"],
    bins=[0, 60, 75, 90, 101],
    labels=["Low", "Medium", "High", "Excellent"],
    right=False
)

attendance_analysis = df.groupby(
    "Attendance_Band", observed=False
)["Average_Marks"].mean()

print("Average marks by attendance band:")
display(attendance_analysis)

# %%
# Question 29: Study hours and average marks

study_analysis = df.groupby("Study_Hours")["Average_Marks"].mean()

print("Average marks according to study hours:")
display(study_analysis)

correlation = df[["Study_Hours", "Average_Marks"]].corr().iloc[0, 1]

print("Correlation:", correlation)


# %%
# Question 30: Internet access comparison

internet_analysis = df.groupby("Internet_Access")["Average_Marks"].mean()

print("Internet access comparison:")
display(internet_analysis)

# %%
# Question 31: NumPy calculations for Python marks

python_marks = df["Python_Marks"].dropna().to_numpy()

print("Median:", np.median(python_marks))

print("Standard Deviation:", np.std(python_marks))

print("First Quartile Q1:", np.percentile(python_marks, 25))

print("Second Quartile Q2:", np.percentile(python_marks, 50))

print("Third Quartile Q3:", np.percentile(python_marks, 75))


# %%
# Question 32: Identify outliers using IQR

Q1 = np.percentile(python_marks, 25)

Q3 = np.percentile(python_marks, 75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR

upper_limit = Q3 + 1.5 * IQR

outliers = python_marks[
    (python_marks < lower_limit) |
    (python_marks > upper_limit)
]

print("Lower limit:", lower_limit)

print("Upper limit:", upper_limit)

print("Potential outliers:", outliers)

# %%
# Question 33: Recommendations for academic support

recommendations = [
    "Provide extra classes for students scoring below 40%.",

    "Encourage students to maintain attendance above 75%.",

    "Arrange additional practice sessions for weak subjects.",

    "Help students create a regular study timetable.",

    "Provide internet facilities and digital learning resources."
]

print("Five Recommendations:")

for number, recommendation in enumerate(recommendations, start=1):
    print(number, recommendation)


