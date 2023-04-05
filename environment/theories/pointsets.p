real : type.

= : [(t : type) -> t -> t -> prop].
!= : [(t : type) -> t -> t -> prop].

set : type.

open : [set -> prop].
closed : [set -> prop].

oint : [real -> real -> set].
cint : [real -> real -> set].

oint_open : [((oint 'a 'b) : set) -> (open (oint 'a 'b))].
cint_closed : [((cint 'a 'b) : set) -> (closed (oint 'a 'b))].

union : [set -> set -> set].
inter : [set -> set -> set].
compl : [set -> set].

diff : [set -> set -> set].
diff_def : [((diff 'a 'b) : set) -> (= (diff 'a 'b) (inter 'a (compl 'b)))].

union_symm : [((union 'a 'b) : set) -> (= (union 'a 'b) (union 'b 'a))].
inter_symm : [((inter 'a 'b) : set) -> (= (inter 'a 'b) (inter 'b 'a))].

u_closed : [(closed 'a) -> (closed 'b) -> (closed (union 'a 'b))].
i_closed : [(closed 'a) -> (closed 'b) -> (closed (inter 'a 'b))].

u_open : [(open 'a) -> (open 'b) -> (open (union 'a 'b))].
i_open : [(open 'a) -> (open 'b) -> (open (inter 'a 'b))].

compl_open : [(open 'a) -> (closed (compl 'a))].
compl_closed : [(closed 'a) -> (open (compl 'a))].

verify union_three_closed_sets {
   let S1 : set. let S2 : set. let S3 : set.
   assume (closed S1). assume (closed S2). assume (closed S3).
   let S : set = (union S1 (union S2 S3)).

   show (closed (union S2 S3)) by u_closed.
   show (closed S) by u_closed.
}

verify diff_closed_union_open {
   let S1 : set. let S2 : set. let S3 : set.
   assume (closed S1). assume (open S2). assume (open S3).
   let S : set = (diff S1 (union S2 S3)).

   show (= (diff S1 (union S2 S3)) (inter S1 (compl (union S2 S3)))) by diff_def.
   show (open (union S2 S3)) by u_open.
   show (closed (compl (union S2 S3))) by compl_open.
   show (closed (inter S1 (compl (union S2 S3)))) by i_closed.
}
