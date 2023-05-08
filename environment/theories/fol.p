/* Minimal first-order logic domain */

/* Type for objects */
object : type.

/* Negation */
not : [prop -> prop].

ex : [[object -> prop] -> prop].
ex_some : [[object -> prop] -> [object -> prop] -> prop].

ex_intro : [('o : object) -> ('p 'o) -> (ex 'p)].

ex_implies : [(ex 'p) -> (implies 'p 'q) -> (ex 'q)].
ex_some_implies_pl : [(ex_some 'p 'q) -> (implies 'q 'r) -> (ex_some 'p 'r)].
ex_some_implies_pr : [(ex_some 'p 'q) -> (implies 'q 'r) -> (ex_some 'r 'p)].

ex_some_implies_ql : [(ex_some 'p 'q) -> (implies 'p 'r) -> (ex_some 'r 'q)].
ex_some_implies_qr : [(ex_some 'p 'q) -> (implies 'p 'r) -> (ex_some 'q 'r)].

implies_trans : [(implies 'p 'q) -> (implies 'q 'r) -> (implies 'p 'r)].

exists_counter_example : [(not (ex 'p)) -> (implies 'q 'p) -> (not (ex 'q))].
exists_none_counter_example_ll : [(not (ex_some 'p 'q)) -> (implies 'r 'q) -> (not (ex_some 'p 'r))].
exists_none_counter_example_lr : [(not (ex_some 'p 'q)) -> (implies 'r 'q) -> (not (ex_some 'r 'p))].
exists_none_counter_example_rl : [(not (ex_some 'p 'q)) -> (implies 'r 'p) -> (not (ex_some 'q 'r))].
exists_none_counter_example_rr : [(not (ex_some 'p 'q)) -> (implies 'r 'p) -> (not (ex_some 'r 'q))].
