import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules, apriori

df = pd.read_excel("online_retail_II.xlsx")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M")

df["month"] = df["InvoiceDate"].dt.month
df["day"] = df["InvoiceDate"].dt.weekday

df["month"].replace(
    [i for i in range(1, 12 + 1)],
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    inplace=True,
)
df["day"].replace(
    [i for i in range(6 + 1)],
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    inplace=True,
)

st.title("Online Retail Dataset Menggunakan algoritma Apriori")


def get_data(month="", day="", country=""):
    data = df.copy()
    filtered = data.loc[
        (data["month"].str.contains(month.title()))
        & (data["day"].str.contains(day.title()))
        & (data["Country"].str.contains(country))
    ]
    return filtered if filtered.shape[0] else "No Result!"


def user_input_features():
    description = st.selectbox("Description", df["Description"].unique())
    month = st.select_slider("Month", df["month"].unique())

    # Periksa apakah nilai default "Saturday" ada dalam kumpulan nilai yang diizinkan
    allowed_days = df["day"].unique()
    default_day = "Saturday" if "Saturday" in allowed_days else allowed_days[0]

    day = st.select_slider("Day", allowed_days)
    country = st.selectbox("Country", df["Country"].unique())

    return month, day, description, country


month, day, description, country = user_input_features()


data = get_data(month=month, day=day, country=country)


def encode(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1


if type(data) != type("No Result"):
    item_count = (
        data.groupby(["Invoice", "Description"])["Description"]
        .count()
        .reset_index(name="Count")
    )
    item_count_pivot = item_count.pivot_table(
        index="Invoice", columns="Description", values="Count", aggfunc="sum"
    ).fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    support = 0.02
    frequent_items = apriori(item_count_pivot, min_support=support, use_colnames=True)

    metric = "lift"
    min_threshold = 1

    rules = association_rules(
        frequent_items, metric=metric, min_threshold=min_threshold
    )[["antecedents", "consequents", "support", "confidence", "lift"]]
    rules.sort_values("confidence", ascending=False, inplace=True)


def parse_list(x):
    x = list(x)
    if len(x) == 1:
        return x[0]
    elif len(x) > 1:
        return ",".join(x)


def return_item_df(item_antecedents):
    data = rules[["antecedents", "consequents"]].copy()

    data["antecedents"] = data["antecedents"].apply(parse_list)
    data["consequents"] = data["consequents"].apply(parse_list)

    if len(data.loc[data["antecedents"] == item_antecedents]) > 0:
        return list(data.loc[data["antecedents"] == item_antecedents].iloc[0, :])
    else:
        return "No Result"


if type(data) != type("No Result"):
    st.markdown("Hasil Rekomendasi : ")
    st.success(
        f"Jika Konsumen Membeli *{description}**, maka membeli **{return_item_df(description)[1]}* secara bersamaan"
    )
