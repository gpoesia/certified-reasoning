action : type.

obligatory : [action -> prop].
permissible : [action -> prop].

person : type.
group : type.
message : type.

event : type.
invite : type.

send_notification : [invite -> action].

accept : [invite -> action].
decline : [invite -> action].

individual_invite : [person -> event -> invite].
group_invite : [group -> event -> invite].

/*  */
verify basic_invite {
       let a : person.
       let b : person.
       let g1 : group.
       let e : event.

       let axiom1 : [('e : event) ->
               (permissible (accept (individual_invite a 'e)))].
       let axiom2 : [('e : event) -> ('g : group) ->
               (permissible (decline (group_invite 'g 'e)))].

       let axiom3 : [('i : invite) ->
                     (permissible (accept 'i)) ->
                     (permissible (send_notification 'i))].

       let i1 : invite = (individual_invite a e).
       let i2 : invite = (group_invite g1 e).

       show (permissible (accept i1)) by axiom1.
       show (permissible (send_notification i1)) by axiom3.
       show (permissible (decline i2)) by axiom2.

}
