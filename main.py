"""
HEART DISEASE PREDICTOR APPLICATION
-------------------------------------------------
Disclaimer: This application was created as the Capstone project for the WGU Computer Science B.S. program.
            It is not intended for any actual medical diagnosis or advice.

This program trains a Logistic Regression machine learning model on Behavioral Risk Factor Surveillance
System (BRFSS) data provided openly by the CDC. It consists of 253,680 samples, which are then reduced to 23,893
after under-sampling the data. After the training event, the ML model is used to make predictions based on
user selected responses to a questionnaire.

The dataset used can be found here:
https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
-------------------------------------------------
This is the main module of the application. It mainly serves as the user interface through the use
of Streamlit (streamlit.io). It consists of two main sections: Descriptive and Predictive.

Descriptive: Shows an analysis of the data, both, before and after under-sampling, along with descriptive
             charts/plots.

Predictive: Provides a questionnaire form that allows a user to assess their risk of heart disease based on
            the learned capacity of the ML model. Results are provided as a percentage (%).
            NOT TO BE CONSIDERED PROFESSIONAL MEDICAL ADVICE OR DIAGNOSIS.

-------------------------------------------------
Author: Felix Peña
C964 | Computer Science Capstone
WGU
"""

# Import libraries
import streamlit as st
import DataDescription as Data
from PIL import Image


# Set font styling for disclaimer text
st.markdown("""
<style>
.disclaimer-text {
    font-size:15px !important;
    font-style: italic;
    color:#0075A2; 
}
</style>
""", unsafe_allow_html=True)


# Function that adjusts page content width
def _max_width_(percent_width: int = 75):
    max_width_str = f"max-width: {percent_width}%;"
    st.markdown(f""" 
                <style> 
                .appview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, unsafe_allow_html=True)


# Import original and under-sampled data from DataDescription
hd = Data.hd_original
hd_under = Data.hd_under

# Load plot images that were created and saved in DataDescription
roc_plot = Image.open("images/roc_curve.png")
conf_plot = Image.open("images/confusion_matrix.png")
corr_plot = Image.open("images/correlation_matrix.png")
age_target_plot = Image.open("images/age_target_plot.png")

# Create main page header
st.title("Heart Disease Analysis and Predictor")
st.write("""
C964 | Computer Science Capstone | WGU
###### Created by Felix A. Peña
""")
st.markdown("""<hr style="height:4px;border:none;color:#333;background-color:#333;" /> """,
            unsafe_allow_html=True)

# Create sidebar and sidebar header
sidebar = st.sidebar
sidebar.header("Data Control")
sidebar.markdown("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
                 unsafe_allow_html=True)

# Sidebar section: Select View and Page Width
data_view = sidebar.selectbox("Select View", ("Descriptive", "Predictive"))
page_width_percent = sidebar.select_slider("Page Width (%)",
                                           options=[50, 65, 75, 85, 90],
                                           value=65,
                                           help="Adjust slider to select page content width in percentage (%).")
_max_width_(page_width_percent)
sidebar.markdown("---")

# If 'Descriptive' view is selected, show the following in main page area
if "Descriptive" in data_view:
    # Display 'Descriptive' view header
    st.header(data_view + " Analysis")
    st.subheader("Understanding our Data and ML Model")
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    # Sidebar section: Raw Data
    sidebar.write('''**Raw Data**''')
    raw_display = sidebar.checkbox("Display Raw Data", value=True)
    raw_display_count = sidebar.number_input("Number of Samples (max: 100)", min_value=1, max_value=100, value=5, step=1)

    # Sidebar section: Value Counts
    sidebar.write('''**Value Counts**''')
    counts_display = sidebar.checkbox("Display Value Counts", value=True)

    # Sidebar section: Classification Report
    sidebar.write('''**Classification Report**''')
    class_rpt = sidebar.checkbox("Display Classification Report", value=True)

    if raw_display:
        st.subheader("""**Raw Data Samples**""")
        st.write(f"Showing {raw_display_count} samples")
        st.markdown("""This table (DataFrame) serves as a look into our original dataset. The dataset consists of 
                    253,680 samples. However, a maximum of the first 200 are available in this view. This table
                    shows the factors taken into question in order to train our ML model to make predictions in future
                    cases. Therefore, these factors reflect the questions that should be asked to new patients. 
                    """)
        st.write(hd.head(raw_display_count))
        st.markdown("---")

    if counts_display:
        st.subheader("""**Value Counts**""")
        st.markdown("""How many samples in our dataset have *heart disease* vs *no heart disease*.<br>
                    Our original dataset was not balanced, meaning that there was a significantly higher number of<br>
                    patients in our dataset without history of heart disease as opposed to those *with* heart<br>
                    disease. By under-sampling our data using `RandomUnderSampler` we are able to provide our ML<br>
                    with more useful training data.<br><br>
                    0.0 = No heart disease<br>
                    1.0 = Heart disease
                    """, True)
        col1, col2 = st.columns(2)
        col1.write("Before under-sampling")
        col1.write(hd.HeartDisease.value_counts())
        col2.write("After under-sampling")
        col2.write(hd_under.HeartDisease.value_counts())
        st.markdown("---")

    if class_rpt:
        st.subheader("""**Classification Report**""")
        st.markdown("""The **classification report** demonstrates a detailed evaluation of our machine learning<br>
                    model based on the training test set and the target predictions.<br>
                    *Ran after under-sampling*.
                    """, True)
        st.table(Data.classification_rpt)
        st.markdown("---")

    # Confusion Matrix section
    st.subheader("""**Confusion Matrix**""")
    st.markdown("""The **confusion matrix** below summarizes the quantities of correct vs incorrect predictions made<br>
                by our Logistic Regression model. As you can see here, it is performing at roughly 3/4 accuracy.<br>
                This performance level is also apparent in our classification report above.<br>
                """, True)
    st.image(conf_plot)
    st.markdown("---")

    # Receiver Operating Characteristic section
    st.subheader("""**Receiver Operating Characteristics**""")
    st.markdown("""Here is the **ROC curve** for our model's performance. The ROC tells us what the relationship<br>
                between **false positive rates** (FPR) and **true positive rates** (TPR) is. In other words, it<br>
                depicts what our model's performance is at all possible classification thresholds.<br>
                """, True)
    st.image(roc_plot)
    st.markdown("---")

    # Age to Target relation section
    st.subheader("""**Age to Target Relation**""")
    st.markdown("""The graph below is an exploration of any potential relation between *age* and *heart disease*.<br>
                The data is split into two target categories: Heart Disease or No Heart Disease<br>
                Then, it is further subdivided by age group within the two major categories.<br><br>
                By analyzing the graph, there appears to be an increased prevalence in heart disease amongst the<br>
                higher age groups in the positive target class, whereas the negative target class shows a normal<br>
                distribution.<br>
                """, True)
    st.image(age_target_plot)
    st.markdown("---")

    # Age to Physical Health Relation section
    st.subheader("""**Relation of Age to Physical Health**""")
    st.markdown("""The graphs below does not explicitly relate to heart disease. However, in an effort to better<br>
                understand factors that may indirectly pertain to our main question of heart disease, it may be<br>
                useful to explore such tangential avenues.<br><br>
                The first belongs to the negative target class (no history of heart disease) while the second graph<br>
                represents the positive target class (history of heart disease).<br><br>
                The Y-axis represents the average number of days in a month (the last month before the patient was<br>
                surveyed) for which poor physical health was experienced, and the X-axis bars represent an<br>
                age group.<br><br>
                **Age Table Key:**<br>
                """, True)

    st.table(Data.age_key)  # Key for ages in the Age to Physical Health graphs

    # Slider that lets users select the age group range to view in the graph
    age_group_selections = st.select_slider("Age Group Range",
                                            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                            value=(1, 13),
                                            help="Adjust both sliders to select an age group range to graph")

    # Call function that creates and saves a graph with the selected age ranges
    Data.age_phys_age_range(age_group_selections[0], age_group_selections[1])

    age_phys_plot_neg = Image.open("images/age_phys_plot0.png")  # Load graph with negative class (No heart disease)
    age_phys_plot_pos = Image.open("images/age_phys_plot1.png")  # Load graph with positive class (Heart disease)
    st.image(age_phys_plot_neg)  # Display negative target class graph
    st.image(age_phys_plot_pos)  # Display positive target class graph
    st.markdown("---")

    # Correlation Matrix section
    st.subheader("""**Correlation Matrix**""")
    st.markdown("""A **Correlation Matrix** tells us what the relation between any two variables (attributes) in<br>
                our data is. The possible range of values is between **-1** and **1**, signifying either a<br>
                negative (indirect) or positive (direct) correlation. In our case, no two attributes have less than<br>
                a -0.4 correlation ratio.
                """, True)
    st.pyplot(Data.corr_plot)
    st.markdown("---")

# If 'Predictive' view is selected, show the following in main page area
if "Predictive" in data_view:
    # Display 'Predictive' view header, sub-header, and disclaimer message
    st.header(data_view + " Analysis")
    st.subheader("Let's check your likelihood of heart disease risk")
    st.markdown("""
    <p class="disclaimer-text">
    DISCLAIMER: THIS WEB-APP DOES NOT PROVIDE MEDICAL ADVICE AND/OR DIAGNOSIS<br>
    The information, including but not limited to, text, graphics, images, and other material contained on this
    web application are for informational<br>purposes only and, additionally, for the specific purpose of the WGU
    computer science Capstone project. No material on this site is intended<br>to be a substitute for professional
    medical advice, diagnosis or treatment. Always seek the advice of your physician or other qualified health<br>
    care provider with any questions you may have regarding a medical condition or treatment and before
    undertaking a new health care regimen.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("""<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.subheader("Heart Disease Prediction")
    st.caption("Fill out the form below and click PREDICT to view your results.")

    # Optional name input field. If name provided, display welcome message
    user_name = st.text_input("First Name (optional)", "First Name")
    if user_name == "First Name" or user_name == "":
        st.write("")
    else:
        st.subheader("Welcome, " + user_name)

    # Create columns to be used by the questionnaire input fields
    col1, col2, col3, col4 = st.columns(4)  # Create 4 columns
    col1b, col2b = st.columns(2)  # Create 2 columns

    # Displayed text options for questionnaire questions
    male_female_display = ('Female', 'Male')
    yes_no_display = ('No', 'Yes')
    diabetes_display = ('No diabetes',
                        'Pre-diabetic or borderline',
                        'Diabetic')
    edu_display = ('Never attended school',
                   'Elementary/Middle',
                   'Some High School',
                   'High School graduate',
                   'Some College',
                   'College graduate')
    income_display = ('Less than $10,000',
                      '$10,000 - $15,000',
                      '$15,000 - $20,000',
                      '$20,000 - $25,000',
                      '$25,000 - $35,000',
                      '$35,000 - $50,000',
                      '$50,000 - $75,000',
                      '$75,000 or more')
    gen_hlth_display = ('1 - Excellent',
                        '2 - Very Good',
                        '3 - Good',
                        '4 - Fair',
                        '5 - Poor')
    bmi_help = '''
    | Weight Level | BMI |
    | ------------ | ------------ |
    | Normal Weight | 18.5 - 25.0 BMI |
    | Underweight | < 18.5 BMI |
    | Overweight: | 25.0 - 30.0 BMI |
    | Obese | > 30.0 BMI |
    '''  # Help message for BMI question to help users understand BMI values

    # Create actual values for each selection of a certain questionnaire question
    # Where the same values can be used for different questions, the options for a previous question may be used
    age_options = list(range(1, len(Data.age_labels) + 1))
    yes_no_options = list(range(len(yes_no_display)))
    diabetes_options = list(range(len(diabetes_display)))
    edu_options = list(range(1, len(edu_display) + 1))
    income_options = list(range(1, len(income_display) + 1))
    gen_hlth_option = list(range(1, len(gen_hlth_display) + 1))

    with col1:
        age = st.selectbox("Age Group", age_options, format_func=lambda x: Data.age_labels[x-1])
        smoker = st.selectbox("Smoked over 100 cigarettes? (lifetime)", yes_no_options,
                              format_func=lambda x: yes_no_display[x])
        high_bp = st.selectbox("History of High Blood Pressure", yes_no_options,
                               format_func=lambda x: yes_no_display[x])

    with col2:
        sex = st.selectbox("Sex", yes_no_options, format_func=lambda x: male_female_display[x])
        diabetes = st.selectbox("Diabetes", diabetes_options, format_func=lambda x: diabetes_display[x])
        high_chol = st.selectbox("History of High Cholesterol", yes_no_options,
                                 format_func=lambda x: yes_no_display[x])

    with col3:
        stroke = st.selectbox("History of stroke", yes_no_options, format_func=lambda x: yes_no_display[x])
        education = st.selectbox("Education Completed", edu_options, format_func=lambda x: edu_display[x-1])
        chol_check = st.selectbox("Checked cholesterol in last 3 yrs?", yes_no_options,
                                  format_func=lambda x: yes_no_display[x])

    with col4:
        bmi = st.number_input("BMI", min_value=0, max_value=150, value=22, help=bmi_help)
        income = st.selectbox("Annual Income", income_options, format_func=lambda x: income_display[x-1])
        diff_walk = st.selectbox("Difficulty walking or climbing stairs?", yes_no_options,
                                 format_func=lambda x: yes_no_display[x])

    fruits = col1b.selectbox("Consume 1 or more fruits per day? (avg)", yes_no_options,
                             format_func=lambda x: yes_no_display[x])

    veggies = col2b.selectbox("Consume 1 or more veggies per day?", yes_no_options,
                              format_func=lambda x: yes_no_display[x])

    st.markdown('---')

    # Begin second portion of questionnaire. If user entered name above, provide progress message.
    if user_name == "First Name" or user_name == "":
        st.write("")
    else:
        st.caption(f"Just a few more questions and we are done, {user_name}")

    # Create another two columns below the divider line for the remaining questions
    col1c, col2c = st.columns(2)

    with col1c:
        st.write("Heavy drinker? (adult men having more than 14 drinks per week and adult women having"
                 "more than 7 drinks per week)")
        hvy_drinker = st.selectbox("Heavy Drinker", yes_no_options, format_func=lambda x: yes_no_display[x])
        st.write("")
        st.write("")

        st.write("Now thinking about your physical health, which includes physical illness and injury,"
                 "for how many days during the past 30 days was your physical health not good?")
        phys_health = st.number_input("Physical Health", min_value=0, max_value=30, value=0)
        st.write("")
        st.write("")

        st.markdown("""
        Would you say that in general your health is:<br>
        1 - Excellent<br>
        2 - Very good<br>
        3 - Good<br>
        4 - Fair<br>
        5 - Poor
        """, unsafe_allow_html=True)
        gen_hlth = st.selectbox("General Health", gen_hlth_option, index=2, format_func=lambda x: gen_hlth_display[x-1])
        st.write("")
        st.write("")

    with col2c:
        st.write("In the past 30 days, have you engaged in physical activity other than your regular job?"
                 "(i.e. gym, running, cycling, etc.)")
        phys_activity = st.selectbox("Physical Activity", yes_no_options, format_func=lambda x: yes_no_display[x])
        st.write("")
        st.write("")

        st.write("Now thinking about your mental health, which includes stress, depression, and problems"
                 "with emotions, for how many days during the past 30 days was your mental health not good?")
        ment_health = st.number_input("Mental Health", min_value=0, max_value=30, value=0)
        st.write("")
        st.write("")

        st.markdown("""
        Do you have any kind of health care coverage, including health insurance, prepaid plans
        or government plans such as, but not limited to:<br>
        HMOs<br>
        Medicare<br>
        TRICARE<br>
        Indian Health Service
        """, unsafe_allow_html=True)
        healthcare = st.selectbox("Health Care Coverage/Plan", yes_no_options, format_func=lambda x: yes_no_display[x])
        st.write("")
        st.write("")

    st.write("Was there a time in the past 12 months when you needed to see a doctor but could\n"
             "not because of cost?")
    med_access = st.selectbox("Medical Accessibility", yes_no_options, format_func=lambda x: yes_no_display[x])

    user_selections = [[high_bp, high_chol, chol_check, bmi, smoker, stroke,
                        diabetes, phys_activity, fruits, veggies, hvy_drinker, healthcare,
                        med_access, gen_hlth, ment_health, phys_health, diff_walk, sex, age,
                        education, income]]

    # Create button and add click action. Action passes user inputs to a function in DataDescription.py
    # which uses the model to return the probability of the user representing the positive target class (heart disease).
    if st.button("PREDICT"):
        user_risk_prob, results = Data.make_prediction(user_selections)
        st.subheader("Your risk is:  " + user_risk_prob.astype(str) + "%")
        # st.table(results)
    else:
        st.write("")
