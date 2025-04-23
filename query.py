import ollama
import psycopg2

# -- PostgreSQL connection config --
DB_CONFIG = {
    "host" : "",        # or your host e.g., '127.0.0.1'
    "database" : "",
    "user" : "",
    "password" : "",
    "port" : 
}

# -- Auto-fetch schema --
def fetch_schema():
    schema_text = ""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE';
        """)
        tables = cur.fetchall()

        for (table,) in tables:
            cur.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table,))
            columns = cur.fetchall()
            column_list = ", ".join([f"{col} {dtype.upper()}" for col, dtype in columns])
            schema_text += f"- {table}({column_list})\n"

        cur.close()
        conn.close()
    except Exception as e:
        print(" Error fetching schema:", e)
        return ""

    return "Tables:\n" + schema_text.strip()

# -- Prompt builder with safety instructions --
def build_prompt(question, schema):
    return f"""
You are a PostgreSQL SQL expert.

ONLY generate a valid, safe **SELECT** SQL query based on the schema below and the user's question.

DO NOT generate queries that modify data.
DO NOT use made-up column names.
Use **only the exact column names** as given in the schema, including their case and spacing.
Always wrap column names in **double quotes** (e.g., "Transaction Type", "Account Name") to preserve exact names.
DO NOT rename columns or change them to snake_case or lowercase.

DO NOT include SQL labels or explanations — only output the raw query.

The column names are "Date","description","amount","Transaction Type","category","Account Name" use this exact case and spacing.
do not add under_score or change the case of the column names.
Here are sample rows to help you understand the data:
"Date"	"description"	"amount"	"Transaction Type"	"category"	"Account Name"
01/01/2018	Amazon	11.11	debit	Shopping	Platinum Card
01/02/2018	Mortgage Payment	1247.44	debit	Mortgage & Rent	Checking
01/02/2018	Thai Restaurant	24.22	debit	Restaurants	Silver Card
01/03/2018	Credit Card Payment	2298.09	credit	Credit Card Payment	Platinum Card
01/04/2018	Netflix	11.76	debit	Movies & DVDs	Platinum Card

Schema:
{schema}

Question: {question}
SQL:
"""




# -- Safety check for generated SQL --
def is_select_query(sql):
    safe = sql.strip().lower().startswith("select") or sql.strip().lower().startswith("with")
    forbidden_keywords = ["insert", "update", "delete", "drop", "alter", "truncate"]
    if any(keyword in sql.lower() for keyword in forbidden_keywords):
        return False
    return safe

# -- Generate SQL using Ollama + Mistral --
def get_sql_from_question(question, schema):
    prompt = build_prompt(question, schema)
    response = ollama.chat(model='mistral', messages=[
        {'role': 'user', 'content': prompt}
    ])
    raw_sql = response['message']['content'].strip()

    # Cleanup: remove any label prefix like 'SQL:', 'SECTION SQL;', etc.
    lines = raw_sql.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().lower().startswith(('sql:', 'section', 'note'))]
    cleaned_sql = "\n".join(cleaned_lines).strip()

    return cleaned_sql


# -- Execute SQL safely --
def execute_sql(sql_query):
    if not is_select_query(sql_query):
        print("Unsafe query detected. Only SELECT statements are allowed.")
        return

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()

        if not rows:
            print("Query ran successfully. No rows returned.")
        else:
            for row in rows:
                print(row)

        cur.close()
        conn.close()
    except Exception as e:
        print(" Error executing query:", e)

# -- Main Loop --
if __name__ == "__main__":
    schema = fetch_schema()
    if not schema:
        print("Could not load schema. Exiting.")
        exit()

    while True:
        question = input("\nAsk a database question (or type 'exit'): ")
        if question.lower() in ["exit", "quit","q"]:
            break

        sql = get_sql_from_question(question, schema)
        print("\n Generated SQL:\n", sql)

        print("\n Validating and executing SQL...")
        execute_sql(sql)
