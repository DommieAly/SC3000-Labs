% Exercise 1.2: Research Data Sharing
% Query: violates_policy(anna).

research_group(visionlab).
dataset(medset).
owns(visionlab, medset).
contains_patient_data(medset).
intern(anna, visionlab).
uploaded_to_public_repo(anna, medset).

sensitive(D) :- dataset(D), contains_patient_data(D).
not_authorized(X, D) :- intern(X, G), owns(G, D), sensitive(D).
publishes(X, D) :- uploaded_to_public_repo(X, D).
violates_policy(X) :- publishes(X, D), sensitive(D), not_authorized(X, D).
