# Configuration and parameters

drop_columns=['Daily20MPractice', 'ReferralSource','Location','DoB']
textual_columns = ['PersonalHistory', 'Motivation']
categorical_columns = ['CourseType', 'Gender', 'Ethnicity']
datetime_columns = ['Days_Since_EarliestCourse', 'AgeAtCourse']
non_textual = categorical_columns + datetime_columns
to_drop = ['Gender', 'CourseType', 'Ethnicity']
categorical_columns = [column for column in categorical_columns if column not in to_drop]

topics_per_column = {
    'PersonalHistory': 3,
    'Motivation': 3,
}

col1a='remainder__PersonalHistory_Topic0'
col2a='remainder__Motivation_Topic0'
col1b='remainder__PersonalHistory_Topic1'
col2b='remainder__Motivation_Topic1'
col1c='remainder__PersonalHistory_Topic2'
col2c='remainder__Motivation_Topic2'

column_pairs = [
    (col1a, col2a),
    (col1a, col2b),
    (col1a, col2c),
    (col1b, col2a),
    (col1b, col2b),
    (col1b, col2c),
    (col1c, col2a),
    (col1c, col2b),
    (col1c, col2c),
]
