import models
import pandas as pd

HEADER = "Original Model IT"
READ_FILE = "test_sentences3.csv"
WRITE_FILE = "test_sentences4.csv"

base_sentences = [
    "Yellowstone National Park was established by the US government in 1972 as the world's first legislated effort at nature conservation.",
    "Taylor reportedly consumed copious amounts of raw fruit and iced milk while attending holiday celebrations during a fundraising event at the Washington Monument.",
    "On July 9, 1937, fire gutted a film storage facility in Little Ferry, New Jersey, rented by the American studio 20th Century Fox.",
    "After nine episodes had aired, the network placed the show on hiatus before it was eventually canceled.",
]

df = pd.read_csv(READ_FILE, sep="\t", header=0, engine="python")

# Function to insert row in the dataframe
# Not sure if needed (model may produce same number of alternatives) but for just in case
def Insert_row(row_number, df, row_value):
    start_upper = 0
    end_upper = row_number
    start_lower = row_number
    end_lower = df.shape[0]
    upper_half = [*range(start_upper, end_upper, 1)]
    lower_half = [*range(start_lower, end_lower, 1)]
    lower_half = [x.__add__(1) for x in lower_half]
    index_ = upper_half + lower_half
    df.index = index_
    df.loc[row_number] = row_value
    df = df.sort_index()
    return df


df[HEADER] = ""
index = 0
for i in range(len(base_sentences)):
    sentence = base_sentences[i]
    alts = models.generate_alternatives(sentence)["alternatives"]
    flat_alts = [item for sublist in alts for item in sublist]
    for j in range(len(flat_alts)):
        if df["Sentences"][index] == sentence:
            df[HEADER][index] = flat_alts[j]
        # Check just in case if more than normal number of alternatives generated.
        else:
            df = Insert_row(index, df, [sentence])
            df[HEADER][index] = flat_alts[j]
        index += 1

df.to_csv(WRITE_FILE, sep="\t")
