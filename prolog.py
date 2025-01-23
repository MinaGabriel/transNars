from pyswip import Prolog

# Initialize Prolog
prolog = Prolog()

# Load the Prolog file
prolog.consult("./logic/fb15k237.pl")

# Query for all countries except Mexico
for result in prolog.query("country(X), X \\= 'mexico'"):
    print(f"country: {result['X']}")
