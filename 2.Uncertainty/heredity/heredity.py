import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Initialise joint probability
    p = 1
    for person in people:
        has_trait = person in have_trait
        n_genes = genes_of_person(person, one_gene, two_genes)

        if not has_parents(person, people):
            # No parental information. Use unconditional probability
            p = p * PROBS['gene'][n_genes] * PROBS['trait'][n_genes][has_trait]
        else:
            # Parental information provided. Use conditional probability
            n_genes_mother = genes_of_person(people[person]['mother'], one_gene, two_genes)
            n_genes_father = genes_of_person(people[person]['father'], one_gene, two_genes)

            # Inherit 0 copies iff both parents don't pass any copy
            if n_genes == 0:
                p_inherit = inheritance_probability(n_genes_mother, False) * inheritance_probability(n_genes_father, False)

            # Inherits 1 copy from mother and 0 from father or viceversa
            elif n_genes == 1:
                p_inherit = (inheritance_probability(n_genes_mother, True) * inheritance_probability(n_genes_father, False)
                             + inheritance_probability(n_genes_mother, False) * inheritance_probability(n_genes_father, True))

            # Inherits 1 copy from each parent
            elif n_genes == 2:
                p_inherit = inheritance_probability(n_genes_mother, True) * inheritance_probability(n_genes_father, True)

            # Add probability of having the trait
            p = p * p_inherit * PROBS['trait'][n_genes][has_trait]

    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        # Update gene
        n_genes = genes_of_person(person, one_gene, two_genes)
        probabilities[person]["gene"][n_genes] += p
        
        # Update trait
        probabilities[person]["trait"][person in have_trait] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        gene_sum = sum(probabilities[person]["gene"].values())
        trait_sum = sum(probabilities[person]["trait"].values())

        # Normalize genes
        for gene, value in probabilities[person]["gene"].items():
            probabilities[person]["gene"][gene] = value / gene_sum

        # Normalize traits
        for trait, value in probabilities[person]["trait"].items():
            probabilities[person]["trait"][trait] = value / trait_sum


def has_parents(person, people):
    """
    Check if a person in people has parents
    """
    parents = [people[person]['mother'], people[person]['father']]
    if any(parents):
        return True
    else:
        return False


def genes_of_person(person, one_gene, two_gene):
    """
    Get the number of genes for the given person
    """
    if person in one_gene:
        return 1
    elif person in two_gene:
        return 2
    else:
        return 0


def inheritance_probability(n_genes_parent, is_inherited):
    """
    Compute and return the probability of passing the gene to the child.

    is_inherited represents whether the child inherits gene or not 
    """
    # A parent that has no copy of the gene can only pass it via mutation
    if n_genes_parent == 0:
        if is_inherited:
            return PROBS['mutation']
        else:
            return 1 - PROBS['mutation']

    # A parent that has 1 copy of the gene has 50% chances to pass it
    elif n_genes_parent == 1:
        return 0.5
    
    # A parent that has 2 copies of the gene always passes it to the child
    elif n_genes_parent == 2:
        if is_inherited:
            return 1 - PROBS['mutation']
        else:
            return PROBS['mutation']

if __name__ == "__main__":
    main()
