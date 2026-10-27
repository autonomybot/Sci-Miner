"""Step 3: load the configured EDGES_CSV_PATH edge list into Neo4j.
Connection details and file paths live in api_config.py."""

import pandas as pd

from api_config import EDGES_CSV_PATH, get_neo4j_driver

# relationship -> (source label, target label). Relationship types cannot be
# Cypher parameters, so they are whitelisted here and inserted into the query.
# "contains" targets are either Variable or Constant; they get their label
# from the hasVariable / hasConstant edges, so no label is set for them here.
SCHEMA = {
    "has": ("System", "File"),
    "contains": ("File", None),
    "encodes": ("File", "Equation"),
    "hasVariable": ("Variable", "Equation"),
    "hasConstant": ("Constant", "Equation"),
}


def upload_edge(tx, source, target, rel):
    src_label, tgt_label = SCHEMA[rel]
    # Merge on name only, then add the label, so a node first created by a
    # "contains" edge (no label yet) is reused rather than duplicated.
    query = "MERGE (s {name: $source}) SET s:" + src_label + " MERGE (t {name: $target}) "
    if tgt_label:
        query += "SET t:" + tgt_label + " "
    tx.run(query + f"MERGE (s)-[:{rel}]->(t)", source=source, target=target)


def main():
    df = pd.read_csv(EDGES_CSV_PATH).fillna("")
    df.columns = [c.lower() for c in df.columns]  # source,target,relationship
    driver = get_neo4j_driver()
    try:
        with driver.session() as session:
            # MERGE adds to the existing graph; edges absent from this CSV are
            # not removed. Each accepted row is committed in its own transaction.
            for _, row in df.iterrows():
                rel = row["relationship"]
                if rel not in SCHEMA or not row["source"] or not row["target"]:
                    continue
                session.execute_write(upload_edge, row["source"], row["target"], rel)
        # This count includes input rows skipped by the validation above.
        print(f"Uploaded {len(df)} edges to Neo4j.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
