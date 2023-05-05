action : type.
event : type.
entity : type.
person : entity.
group : entity.
message : type.


/* base deontic types */
obligatory : [action -> prop].
permissible : [action -> prop].
impermissible : [action -> prop].
/* base axioms */
ob_perm : [(obligatory 'a) -> (permissible 'a)].
im_perm : [(impermissible 'a) -> (not (permissible 'a))].



invite : type.
individual_invite : [person -> event -> invite].
group_invite : [group -> event -> invite].
send_notification : [invite -> action].
accept : [invite -> action].
decline : [invite -> action].
cancel_event : [event -> action].

reminder : type.
none : [person -> event -> reminder].
minutes_before : [person -> event -> reminder].
hours_before : [person -> event -> reminder].
days_before : [person -> event -> reminder].
set_reminder : [reminder -> action].


duration : type.
short : [event -> prop].
long : [event -> prop].

priority: type.
high : [person -> event -> prop].
low : [person -> event -> prop].
/* event_has_priority : [person -> event -> priority -> prop].*/


recurrence : type.
daily : [event -> prop].
weekly : [event -> prop].
monthly : [event -> prop].
yearly : [event -> prop].

availability :type.
busy : [person -> event -> prop].
free : [person -> event -> prop].
tentative : [person -> event -> prop].
check_availability : [person -> event -> action].
/*event_has_avail : [person -> event -> availability -> prop].*/

category : type.
meeting : [event -> prop].
conference : [event -> prop].
social : [event -> prop].
personal : [event -> prop].

visibility : type.
public : [event -> prop].
private : [event -> prop].
confidential : [event -> prop].
change_visibility : [event -> [event -> prop] -> action].


organizer : [event -> person -> prop].
participant : [event -> person -> prop].
group_participant : [event -> group -> prop].

add_participant : [event -> entity -> action].
remove_participant : [event -> entity -> action].
delegate_event : [event -> person -> action].

update_event : [event -> [event -> prop] -> action].
reschedule_event : [event -> [event -> prop] -> action].
request_event_update : [person -> event -> action].
suggest_alternative_time : [person -> event -> action].

/*
verify calendar_management {
let a1 : person.
let a2 : person.
let a3 : person.

let g1 : group.
let g2 : group.

let e1 : event.
let e2 : event.
let e3 : event.
let e4 : event.
let e5 : event.

let l1 : location.
let l2 : location.

let p1 : priority.
let p2 : priority.

let r1 : reminder.
let r2 : reminder.

let c1 : category.
let c2 : category.

let axiom1 : [('e : event) -> (permissible (accept (individual_invite a1 'e)))].  
let axiom2 : [('e : event) -> ('g : group) -> (permissible (decline (group_invite 'g 'e)))].  
let axiom3 : [('e : event) -> (permissible (add_participant 'e a2 'e))].  
let axiom4 : [('e : event) -> ('a : person) -> (impermissible (remove_participant 'e 'a 'e))].  
let axiom5 : [('ea : event) -> ('eb : event) -> (obligatory (update_event 'ea 'eb 'ea))].  
let axiom6 : [('e : event) -> (optional (change_visibility 'e private 'e))].  
let axiom7 : [('e : event) -> ('r : reminder) -> (notoptional (set_reminder 'e 'r 'e))].  
let axiom8 : [('e : event) -> (obligatory (cancel_event 'e)) -> (impermissible (reschedule_event 'e short 'e))].  
let axiom9 : [('e : event) -> (permissible (request_event_update a3 e4))].  
let axiom10 : [('e : event) -> (tentative a1 'e) -> (optional (delegate_event 'e a2))]. 

let i1 : invite = (individual_invite a1 e1).
let i2 : invite = (group_invite g1 e2).
let l1 : location.
let p1 : high.

show (permissible (accept i1)) by axiom1.

show (permissible (decline i2)) by axiom2.

show (permissible (add_participant e1 a2 e1)) by axiom3.

let a4 : person.
let e6 : event.


show (impermissible (remove_participant e6 a4 e6)) by axiom4.

show (obligatory (update_event e2 e3 e2)) by axiom5.

show (optional (change_visibility e3 private e3)) by axiom6.

let e7 : event.
show (notoptional (set_reminder e7 r1 e7)) by axiom7.

let e8 : event.
assume (obligatory (cancel_event e8)).
show (impermissible (reschedule_event e8 short e8)) by axiom8.

show (permissible (request_event_update a3 e4)) by axiom9.

let e9 : event.
assume (tentative a1 e9).

show (optional (delegate_event e9 a2)) by axiom10.

}


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


*/