% Exercise 2.2: Clause order and cyclic dependencies
% Queries: requires_a(visionapp, driver).
%          requires_b(visionapp, driver).

depends_on(visionapp, pytorch).
depends_on(visionapp, opencv).
depends_on(pytorch, cuda).
depends_on(opencv, numpy).
depends_on(cuda, driver).
% depends_on(driver, pytorch).   % uncomment for the cyclic dependency part

% Version A
requires_a(X, Y) :- depends_on(X, Y).
requires_a(X, Y) :- depends_on(X, Z), requires_a(Z, Y).

% Version B
requires_b(X, Y) :- depends_on(X, Z), requires_b(Z, Y).
requires_b(X, Y) :- depends_on(X, Y).
