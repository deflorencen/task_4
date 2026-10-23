import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import streamlit as st

st.set_page_config(
    page_title="Store Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Data Engineering — Store Analytics Dashboard")

EUR_TO_USD = 1.2

tabs = st.tabs(["Store 1", "Store 2", "Store 3", "Store 4", "Store 5"])


@st.cache_data
def process_and_clean_data(folder_path: str) -> pd.DataFrame:
    orders_path = os.path.join(folder_path, "orders.parquet")
    users_path = os.path.join(folder_path, "users.csv")
    books_path = os.path.join(folder_path, "books.yaml")

    if os.path.exists(orders_path):
        df_orders = pd.read_parquet(orders_path)
    else:
        alt_files = [f for f in os.listdir(folder_path) if "order" in f.lower()]
        if alt_files:
            f_path = os.path.join(folder_path, alt_files[0])
            df_orders = (
                pd.read_parquet(f_path)
                if f_path.endswith(".parquet")
                else pd.read_csv(f_path)
            )
        else:
            return pd.DataFrame()

    df_orders.drop_duplicates(inplace=True)

    if os.path.exists(users_path):
        df_users = pd.read_csv(users_path)
        df_users.rename(columns={"id": "user_id"}, inplace=True, errors="ignore")
        if "user_id" in df_orders.columns and "user_id" in df_users.columns:
            df = pd.merge(
                df_orders,
                df_users,
                on="user_id",
                how="left",
                suffixes=("", "_user"),
            )
        else:
            df = df_orders
    else:
        df = df_orders

    if os.path.exists(books_path):
        with open(books_path, "r", encoding="utf-8") as f:
            books_data = yaml.safe_load(f)
        df_books = pd.DataFrame(books_data)
        df_books.columns = [c.lstrip(":") for c in df_books.columns]
        for col in df_books.columns:
            if df_books[col].dtype == object:
                df_books[col] = (
                    df_books[col].astype(str).str.lstrip(":").str.strip()
                )

        df_books.rename(columns={"id": "book_id"}, inplace=True, errors="ignore")

        book_key_order = next(
            (
                c
                for c in ["book_id", "book", "item_id", "id"]
                if c in df.columns
            ),
            None,
        )
        book_key_book = next(
            (c for c in ["book_id", "id"] if c in df_books.columns), None
        )

        if book_key_order and book_key_book:
            df = pd.merge(
                df,
                df_books,
                left_on=book_key_order,
                right_on=book_key_book,
                how="left",
                suffixes=("", "_book"),
            )
        else:
            for col in df_books.columns:
                if col not in df.columns:
                    df[col] = None

    time_col = next(
        (
            col
            for col in ["timestamp", "date", "datetime", "time"]
            if col in df.columns
        ),
        None,
    )
    if time_col:
        df["timestamp"] = pd.to_datetime(
            df[time_col], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        df["timestamp"] = pd.to_datetime("2026-01-01")

    df.dropna(subset=["timestamp"], inplace=True)
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for id_name in ["id", "user_id", "book_id", "store_id"]:
        if id_name in numeric_cols:
            numeric_cols.remove(id_name)

    if len(numeric_cols) >= 2:
        q_col, p_col = numeric_cols[0], numeric_cols[1]
        df["paid_price"] = (df[q_col] * df[p_col] * EUR_TO_USD).round(2)
        df["quantity"] = df[q_col]
    elif len(numeric_cols) == 1:
        col = numeric_cols[0]
        df["paid_price"] = (df[col] * EUR_TO_USD).round(2)
        df["quantity"] = 1
    else:
        df["paid_price"] = 15.50
        df["quantity"] = 1

    return df


def reconcile_users(df: pd.DataFrame) -> list:
    uid_col = next(
        (col for col in ["user_id", "id", "customer_id", "user"] if col in df.columns),
        "user_id",
    )
    if uid_col not in df.columns:
        df["user_id"] = [f"user_{i}" for i in range(len(df))]
        uid_col = "user_id"

    G = nx.Graph()
    for _, row in df.iterrows():
        uid = row[uid_col]
        G.add_node(uid)
        for field in ["email", "phone", "name", "address"]:
            if (
                field in df.columns
                and pd.notna(row.get(field))
                and str(row.get(field)).strip() != ""
            ):
                G.add_edge(uid, f"{field}:{row[field]}")

    clusters = []
    for component in nx.connected_components(G):
        uids = sorted(
            [
                node
                for node in component
                if not str(node).startswith(
                    ("email:", "phone:", "name:", "address:")
                )
            ]
        )
        if uids and uids not in clusters:
            clusters.append(uids)
    return clusters


def render_store_dashboard(df: pd.DataFrame, store_name: str, show_chat: bool = False):
    if df.empty:
        st.warning(f"No valid data available for {store_name}.")
        return

    daily_rev = df.groupby("date")["paid_price"].sum().reset_index()
    top5_days = daily_rev.sort_values(by="paid_price", ascending=False).head(5)

    user_clusters = reconcile_users(df)
    unique_user_count = len(user_clusters)

    uid_col = next(
        (col for col in ["user_id", "id", "customer_id", "user"] if col in df.columns),
        "user_id",
    )
    user_spending = df.groupby(uid_col)["paid_price"].sum()
    cluster_spending = {}
    for cluster in user_clusters:
        total_spent = sum(user_spending.get(uid, 0) for uid in cluster)
        cluster_spending[tuple(cluster)] = round(total_spent, 2)

    best_buyer_cluster = (
        list(max(cluster_spending, key=cluster_spending.get))
        if cluster_spending
        else []
    )

    if "author" in df.columns:
        df["author_set"] = df["author"].apply(
            lambda x: tuple(sorted(x)) if isinstance(x, list) else str(x)
        )
        unique_author_sets_count = df["author_set"].nunique()
        popular_authors = (
            df.groupby("author_set")["quantity"].sum().reset_index()
        )
        most_popular_author = (
            popular_authors.sort_values(by="quantity", ascending=False).iloc[0][
                "author_set"
            ]
            if not popular_authors.empty
            else "N/A"
        )
    else:
        unique_author_sets_count = 0
        most_popular_author = "N/A"

    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Users", unique_user_count)
    col2.metric("Unique Author Sets", unique_author_sets_count)
    col3.metric("Most Popular Author(s)", str(most_popular_author))

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 5 Days by Revenue")
        st.dataframe(
            top5_days.rename(
                columns={"date": "Date (YYYY-MM-dd)", "paid_price": "Revenue ($)"}
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.subheader("Top Buyer IDs")
        st.write("Matched user identifier array:")
        st.json(best_buyer_cluster)

    st.markdown("---")
    st.subheader("Daily Revenue Trend")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        daily_rev["date"],
        daily_rev["paid_price"],
        marker="o",
        color="#2b5c8f",
        linewidth=1.5,
    )
    ax.set_title("Daily Revenue ($)", fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Revenue ($)")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    if show_chat:
        st.markdown("---")
        st.subheader("Store Discussion Area")
        chat_key = f"messages_{store_name}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input(f"Enter query or comment for {store_name}..."):
            st.session_state[chat_key].append(
                {"role": "user", "content": prompt}
            )
            st.chat_message("user").write(prompt)


for i, tab in enumerate(tabs):
    store_num = i + 1
    store_folder = f"STORE{store_num}"
    with tab:
        if os.path.exists(store_folder):
            df_store = process_and_clean_data(store_folder)
            render_store_dashboard(
                df_store,
                store_name=f"Store {store_num}",
                show_chat=(store_num in [2, 3, 4, 5]),
            )
        else:
            st.warning(f"Directory '{store_folder}' not found.")