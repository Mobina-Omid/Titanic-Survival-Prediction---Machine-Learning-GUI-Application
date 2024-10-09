# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Load the Titanic dataset
df = pd.read_csv('DataSet/titanic.csv')

# Check DataSet
print(df.head().to_string())

print(df.info())

print(df.describe())

print(df.shape)

# Data Preparation
missing_values = df.isnull().sum()
# print(missing_values)

df['age'] = df['age'].fillna(df['age'].median()).round().astype(int)

df.drop(['ticket', 'cabin', 'name', 'home.dest', 'body', 'boat', 'embarked'], axis=1, inplace=True)

df.dropna(subset=['fare', 'pclass', 'survived', 'sex', 'sibsp', 'parch'], inplace=True)

missing_values = df.isnull().sum()
# print(missing_values)
# print(df.shape)

duplicates = df.duplicated()

# print(f"Number of duplicate rows: {duplicates.sum()}")

# if duplicates.sum() > 0:
#     print(df[duplicates])

df.drop_duplicates(inplace=True)

# visualization
plots_info = [

    ('sex', 'bar'),
    ('age', 'hist'),
    ('pclass', 'bar'),
    ('fare', 'hist'),
    ('sibsp', 'bar'),
    ('parch', 'bar')
]

plt.figure(figsize=(10, 6))

for i, (column, plot_type) in enumerate(plots_info, 1):
    plt.subplot(2, 3, i)

    if plot_type == 'bar':
        sns.barplot(x=column, y='survived', data=df)
    elif plot_type == 'hist':
        sns.histplot(x=column, hue='survived', data=df, bins=10, kde=False)

    plt.title(f'Survival rate by {column}')

plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(df['fare'])
plt.title('Boxplot for Fare (checking for outliers)')
plt.show()

plt.show()

df = df[df['fare'] <= 300]

df['sex'] = df['sex'].map({'male': 0, 'female': 1})

df['family_size'] = df['sibsp'] + df['parch']
df['is_alone'] = (df['family_size'] == 0).astype(int)

age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]
age_labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

fare_bins = [0, 50, 100, 150, 200, 250, 300]
fare_labels = ['0-50', '51-100', '101-150', '151-200', '201-250', '251-300']
df['fare_group'] = pd.cut(df['fare'], bins=fare_bins, labels=fare_labels, right=False)

features_to_group_by = ['age_group', 'sex', 'pclass', 'fare', 'family_size']
survival_rate_columns = ['age_group_survival_rate', 'sex_survival_rate', 'pclass_survival_rate',
                         'fare_survival_rate', 'family_size_survival_rate']

survival_rates = {}

# Calculate survival rates for each feature group and map them to the DataFrame
for feature, rate_column in zip(features_to_group_by, survival_rate_columns):
    if feature == 'fare':
        fare_bins = pd.qcut(df['fare'], q=4)
        survival_rates[rate_column] = df.groupby(fare_bins, observed=False)['survived'].mean() * 100
        df[rate_column] = pd.cut(df['fare'], bins=survival_rates[rate_column].index.categories).map(
            survival_rates[rate_column])
    else:
        survival_rates[rate_column] = df.groupby(feature, observed=False)['survived'].mean() * 100
        df[rate_column] = df[feature].map(survival_rates[rate_column])

survival_by_age_group = df.groupby('age_group', observed=False)['survived'].mean() * 100
survival_by_age_group = survival_by_age_group.reset_index()

# Define the features and titles for plotting
plot_features = [
    ('age', 'Age', 'Age Distribution of Passengers', 'Survival Rate by Age Group', 'age_group', 'Age Group'),
    ('sex', 'Sex', 'Sex Distribution of Passengers', 'Survival Rate by Sex', 'sex', 'Sex'),
    ('fare', 'Fare', 'Fare Distribution of Passengers', 'Survival Rate by Fare Group', 'fare_group', 'Fare Group'),
    ('family_size', 'Family_Size', 'Family_Size Distribution of Passengers', 'Survival Rate by Family_Size', 'family_size',
     'Family_Size'),
    ('pclass', 'Passenger Class', 'Passenger Class Distribution', 'Survival Rate by Pclass', 'pclass', 'Pclass')
]

for feature, xlabel, dist_title, rate_title, group_by_col, group_label in plot_features:
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    if feature in ['age', 'fare']:
        sns.histplot(df[feature], bins=30, kde=True)
    else:
        sns.countplot(x=feature, data=df)
        if feature == 'sex':
            plt.xticks([0, 1], ['Male', 'Female'])

    plt.title(dist_title)
    plt.xlabel(xlabel)
    plt.ylabel('Number of Passengers')

    plt.subplot(1, 2, 2)
    if feature in ['age', 'fare']:
        grouped_data = df.groupby(group_by_col, observed=False)['survived'].mean() * 100
        grouped_data = grouped_data.reset_index()
        ax = sns.barplot(x=group_by_col, y='survived', data=grouped_data)
    else:
        grouped_data = df.groupby(feature, observed=False)['survived'].mean() * 100
        grouped_data = grouped_data.reset_index()
        ax = sns.barplot(x=feature, y='survived', data=grouped_data)
        if feature == 'sex':
            plt.xticks([0, 1], ['Male', 'Female'])  # Label for sex survival rate

    plt.ylabel('Survival Percentage (%)')
    plt.title(rate_title)

    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.tight_layout()
    plt.show()

#  Regression Model based on raw data
X = df[['sex', 'age', 'pclass', 'fare', 'sibsp', 'parch']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_accuracyRD = 0
best_degreeRD = 0

for degreeRD in range(1, 11):
    poly = PolynomialFeatures(degree=degreeRD)
    X_train_polyRD = poly.fit_transform(X_train)
    X_test_polyRD = poly.transform(X_test)

    scalerRD = StandardScaler()
    X_train_poly_scaledRD = scalerRD.fit_transform(X_train_polyRD)
    X_test_poly_scaledRD = scalerRD.transform(X_test_polyRD)

    modelRD = LogisticRegression(max_iter=10000)
    modelRD.fit(X_train_poly_scaledRD, y_train)

    y_predRD = modelRD.predict(X_test_poly_scaledRD)

    accuracyRD = accuracy_score(y_test, y_predRD)

    print(f'DegreeRD: {degreeRD}, AccuracyRD: {accuracyRD}')

    if accuracyRD > best_accuracyRD:
        best_accuracyRD = accuracyRD
        best_degreeRD = degreeRD

print(f'Best Degree for Raw Data: {best_degreeRD}, Best Accuracy for Raw Data: {best_accuracyRD}')

poly_bestRD = PolynomialFeatures(degree=best_degreeRD)
X_train_poly_bestRD = poly_bestRD.fit_transform(X_train)
X_test_poly_bestRD = poly_bestRD.transform(X_test)

scalerRD = StandardScaler()
X_train_poly_best_scaledRD = scalerRD.fit_transform(X_train_poly_bestRD)
X_test_poly_best_scaledRD = scalerRD.transform(X_test_poly_bestRD)

final_modelRD = LogisticRegression(max_iter=10000)
final_modelRD.fit(X_train_poly_best_scaledRD, y_train)
final_accuracyRD = accuracy_score(y_test, final_modelRD.predict(X_test_poly_best_scaledRD))

print(f'Final Raw Data Model Accuracy: {final_accuracyRD}')

# print(df[['family_size', 'age_group', 'fare_group']].isnull().sum())
df.dropna(subset=['age_group'], inplace=True)
# print(df[['family_size', 'age_group', 'fare_group']].isnull().sum())

# Regression Model based on Survival Rate
Xsr = df[['age_group_survival_rate', 'sex_survival_rate', 'pclass_survival_rate','fare_survival_rate', 'family_size_survival_rate']]
ysr = df['survived']

X_trainSR, X_testSR, y_trainSR, y_testSR = train_test_split(Xsr, ysr, test_size=0.2, random_state=42)

best_accuracySR = 0
best_degreeSR = 0

for degreeSR in range(1, 11):
    polySR = PolynomialFeatures(degree=degreeSR)
    X_train_polySR = polySR.fit_transform(X_trainSR)
    X_test_polySR = polySR.transform(X_testSR)

    scalerSR = StandardScaler()
    X_train_poly_scaledSR = scalerSR.fit_transform(X_train_polySR)
    X_test_poly_scaledSR = scalerSR.transform(X_test_polySR)

    modelSR = LogisticRegression(max_iter=10000)
    modelSR.fit(X_train_poly_scaledSR, y_trainSR)

    y_predSR = modelSR.predict(X_test_poly_scaledSR)

    accuracySR = accuracy_score(y_testSR, y_predSR)

    print(f'DegreeSR: {degreeSR}, AccuracySR: {accuracySR}')

    if accuracySR > best_accuracySR:
        best_accuracySR = accuracySR
        best_degreeSR = degreeSR

print(f'Best Survival Rate Degree: {best_degreeSR}, Best Survival Rate Accuracy: {best_accuracySR}')

poly_bestSR = PolynomialFeatures(degree=best_degreeSR)
X_train_poly_bestSR = poly_bestSR.fit_transform(X_trainSR)
X_test_poly_bestSR = poly_bestSR.transform(X_testSR)

scalerSR = StandardScaler()
X_train_poly_best_scaledSR = scalerSR.fit_transform(X_train_poly_bestSR)
X_test_poly_best_scaledSR = scalerSR.transform(X_test_poly_bestSR)

final_modelSR = LogisticRegression(max_iter=10000)
final_modelSR.fit(X_train_poly_best_scaledSR, y_trainSR)
final_accuracySR = accuracy_score(y_testSR, final_modelSR.predict(X_test_poly_best_scaledSR))

print(f'Final Survival Rate Model Accuracy: {final_accuracySR}')

# GUI
def predict_survival():
    age = float(entry_fields['Age :'].get())
    sex = gender_var.get()
    pclass = int(entry_fields['Pclass :'].get())
    sibsp = int(entry_fields['Siblings/Spouse (SibSp) :'].get())
    parch = int(entry_fields['Parents/Children (Parch) :'].get())
    fare = float(entry_fields['Fare :'].get())

    input_data = [[pclass, sex, age, sibsp, parch, fare]]

    input_data_scaledRD = scalerRD.transform(input_data)
    input_data_polyRD = poly_bestRD.transform(input_data_scaledRD)

    survival_probability = final_modelRD.predict_proba(input_data_polyRD)[0][1]

    survival_percentage = survival_probability * 100

    messagebox.showinfo("Prediction", f"The model predicts that the person has a {final_accuracy * 100:.2f}% chance of survival.")
def add_placeholder(entry, placeholder):
    entry.insert(0, placeholder)
    entry.bind("<FocusIn>", lambda event: clear_placeholder(entry, placeholder))
    entry.bind("<FocusOut>", lambda event: restore_placeholder(entry, placeholder))

def clear_placeholder(entry, placeholder):
    if entry.get() == placeholder:
        entry.delete(0, 'end')
        entry.config(fg='black')

def restore_placeholder(entry, placeholder):
    if entry.get() == '':
        entry.insert(0, placeholder)
        entry.config(fg='grey')

def predict_survival():
    try:
        age = float(entry_fields['Age :'].get())
        if age < 0 or age > 80:
            raise ValueError("Age must be between 0 and 80.")

        sex = int(gender_var.get())
        if sex not in [0, 1]:
            raise ValueError("Sex must be 0 (Male) or 1 (Female).")

        pclass = int(entry_fields['Pclass :'].get())
        if pclass not in [1, 2, 3]:
            raise ValueError("Pclass must be 1, 2, or 3.")

        sibsp = int(entry_fields['Siblings/Spouse (SibSp) :'].get())
        if sibsp < 0 or sibsp > 8:
            raise ValueError("Siblings/Spouse (SibSp) must be between 0 and 8.")

        parch = int(entry_fields['Parents/Children (Parch) :'].get())
        if parch < 0 or parch > 6:
            raise ValueError("Parents/Children (Parch) must be between 0 and 6.")

        fare = float(entry_fields['Fare :'].get())
        if fare < 0 or fare > 500:
            raise ValueError("Fare must be between 0 and 500.")

        input_data = pd.DataFrame([[sex, age, pclass, fare, sibsp, parch]],
                                  columns=['sex', 'age', 'pclass', 'fare', 'sibsp', 'parch'])
        input_data_polyRD = poly_bestRD.transform(input_data)
        input_data_poly_scaledRD = scalerRD.transform(input_data_polyRD)

        survival_probability = final_modelRD.predict_proba(input_data_poly_scaledRD)[0][1]
        survival_percentage = survival_probability * 100

        messagebox.showinfo("Prediction",
                            f"The model predicts that the person has a {survival_percentage:.2f}% chance of survival.")

    except ValueError as e:
        messagebox.showerror("Invalid Input", str(e))
placeholders = {
    'Age :': ' 0 - 80 ',
    'Pclass :': ' 1, 2 or 3',
    'Siblings/Spouse (SibSp) :': ' 0 - 8',
    'Parents/Children (Parch) :': ' 0 - 9',
    'Fare :': ' 0 - 300 '
}
def create_gender_selection():
    gender_var = tk.IntVar()
    gender_var.set(0)

    canvas.create_text(180, 50, text="Sex :", font=('Arial', 10, 'bold'), fill='black', anchor='e')
    male_radio = tk.Radiobutton(root, text='Male', variable=gender_var, value=0, font=('Arial', 10))
    female_radio = tk.Radiobutton(root, text='Female', variable=gender_var, value=1, font=('Arial', 10))

    canvas.create_window(230, 50, window=male_radio)
    canvas.create_window(300, 50, window=female_radio)

    return gender_var

# GUI Setup
root = tk.Tk()
root.title("Titanic Survival Predictor")
root.iconbitmap('Images/ship_boat_vessel_icon_183225.ico')
root.resizable(0, 0)
root.geometry('350x300')
right = int(root.winfo_screenwidth() / 2 - 350 / 2)
down = int(root.winfo_screenheight() / 2 - 300 / 2)
root.geometry('+{}+{}'.format(right, down))

bg_image = Image.open('Images/ship_boat_vessel_icon_183225.png')  # Replace with your image path
bg_image = bg_image.resize((350, 300), Image.Resampling.LANCZOS)  # Resize to fit the window
bg_photo = ImageTk.PhotoImage(bg_image)

canvas = tk.Canvas(root, width=350, height=300)
canvas.pack(fill='both', expand=True)

canvas.create_image(0, 0, image=bg_photo, anchor='nw')

canvas.create_rectangle(0, 0, 350, 300, fill='#ffffff', stipple='gray50')  # Semi-transparent white

fields = ['Age :', 'Pclass :', 'Siblings/Spouse (SibSp) :', 'Parents/Children (Parch) :', 'Fare :']
entry_fields = {}

gender_var = create_gender_selection()

for i, field in enumerate(fields):
    canvas.create_text(180, 80 + (i * 30), text=field, font=('Arial', 10, 'bold'), fill='black', anchor='e')  # No label widget

    entry = tk.Entry(root, fg='gray')
    add_placeholder(entry, placeholders[field])
    canvas.create_window(260, 80 + (i * 30), window=entry)
    entry_fields[field] = entry

predict_button = tk.Button(root, text='chance of survival', command=predict_survival, font=('Arial', 10, 'bold'),padx=10, pady=5)  # Make the button bigger with padding
canvas.create_window(175, 250, window=predict_button)

root.mainloop()
