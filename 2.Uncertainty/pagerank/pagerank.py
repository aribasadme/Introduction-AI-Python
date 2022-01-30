import os
import random
import re
import sys
import math
import time

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    model = dict()
    n_pages = len(corpus)
    n_links = len(corpus[page])

    if n_links == 0:
        # If a page has no links, it has links to all pages in the corpus, including itself
        # distribute probability equally among all pages
        p = 1 / n_pages
        model = {pg: p for pg in corpus}
        return model
    else:
        p = (1 - damping_factor) / n_pages
        for pg in corpus:
            model[pg] = p
            if pg in corpus[page]:
                model[pg] += damping_factor / n_links
        return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    random.seed()
    pageranks = {page: 0 for page in corpus}
    # Prepare first sample, choose a page equally among all pages
    random_page = random.choice(list(corpus))
    pageranks[random_page] = 1 / n

    # Calculate all other samples
    for i in range(1, n):
        model = transition_model(corpus, random_page, damping_factor)
        # Choose next page using the transition model of the previous sample
        weights = list(model.values())
        random_page = random.choices(list(corpus), weights, k=1)[0]
        pageranks[random_page] += 1 / n

    return pageranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold = 0.001
    variations = {page: math.inf for page in corpus}
    iter = 0
    n_pages = len(corpus)

    # Set initial PageRank values
    pageranks = {page: 1 / n_pages for page in corpus}
    # Constant value of the formula
    const = (1 - damping_factor) / n_pages

    while any(variation >= threshold for variation in variations.values()):
        previous_pageranks = pageranks.copy()
        for page in corpus:
            sigma = 0
            # Find parent pages that link to page
            for link_page, links in corpus.items():
                # Page with no links interpreted as having one link for every page in corpus
                if len(links) == 0:
                    links = set(corpus.keys())
                if page in links:
                    # Compute parent's PageRank
                    n_links = len(links)
                    sigma += previous_pageranks[link_page] / n_links

            pageranks[page] = const + damping_factor * sigma
            # Get variation for each page
            variations[page] = abs(pageranks[page] - previous_pageranks[page]) 
    return pageranks


if __name__ == "__main__":
    main()
