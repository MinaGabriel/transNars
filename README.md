# transNars
Learn Versatile Knowledge Graph Embeddings by Capturing Semantics with Non-Axiomatic Reasoning System (NARS)

[TOC]

NARS IMPROVEMENT 😛


# Data Definitions for Knowledge Graph Embedding

## Relations and their IDs
- **works_at**: ID = 1
- **lives_in**: ID = 2

## Entities and their IDs
- **Alice**: ID = 101
- **Bob**: ID = 102
- **Google**: ID = 201
- **New York**: ID = 202

## Entity Classes
- **Person**: ID = 1
- **Company**: ID = 2
- **City**: ID = 3

## Domain and Range Specifications
- **Domain for "works_at" (ID = 1)**: Persons only (`{1}`)
- **Range for "works_at" (ID = 1)**: Companies only (`{2}`)
- **Domain for "lives_in" (ID = 2)**: Persons only (`{1}`)
- **Range for "lives_in" (ID = 2)**: Cities only (`{3}`)

## Entity Types
- Alice is a Person: `instype_all[101] = {1}`
- Bob is a Person: `instype_all[102] = {1}`
- Google is a Company: `instype_all[201] = {2}`
- New York is a City: `instype_all[202] = {3}`

## Example Negative Batch
Consider the following triples that do not adhere to domain and range specifications:

- Alice works at New York? `[101, 1, 202]`
- Bob lives in Google? `[102, 2, 201]`

## Checking Conformity
### Triple 1: `[101, 1, 202]`
- **Domain check**: Is the domain `{1}` for the relation "works_at" applicable to Alice's type `{1}`?
  - **Result**: Yes, since Alice is a Person.
- **Range check**: Is the range `{2}` for "works_at" applicable to New York's type `{3}`?
  - **Result**: No, the relation expects a Company, but New York is a City.
  - **Overall**: This triple does not conform to the schema. Label = 0

### Triple 2: `[102, 2, 201]`
- **Domain check**: Is the domain `{1}` for "lives_in" applicable to Bob's type `{1}`?
  - **Result**: Yes, since Bob is a Person.
- **Range check**: Is the range `{3}` for "lives_in" applicable to Google's type `{2}`?
  - **Result**: No, the relation expects a City, but Google is a Company.
  - **Overall**: This triple also does not conform to the schema. Label = 0

## Output
The negative batch evaluation results in labels indicating non-conformity for both triples:
- **Labels**: `[0, 0]`

This indicates that neither of the sampled negative triples meets the schema requirements set forth by their respective relations' domain and range constraints.
