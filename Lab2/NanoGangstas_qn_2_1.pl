% Exercise 2.1: Software Dependency Reasoning
% Queries: requires(visionapp, P).        % press ; to list every package
%          requires(visionapp, driver).

depends_on(visionapp, pytorch).
depends_on(visionapp, opencv).
depends_on(pytorch, cuda).
depends_on(opencv, numpy).
depends_on(cuda, driver).

requires(X, Y) :- depends_on(X, Y).
requires(X, Y) :- depends_on(X, Z), requires(Z, Y).
