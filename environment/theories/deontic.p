action : type.
event : type.
person : type.
group : type.
message : type.
invite : type.

obligatory : [action -> prop].
permissible : [action -> prop].
optional: [action -> prop].
omissible : [action -> prop].
impermissible : [action -> prop].
notoptional: [action -> prop].

duration : type.
short : duration.
long : duration.

location : type.

priority : type.
high : priority.
low : priority.

send_notification : [invite -> action].

accept : [invite -> action].
decline : [invite -> action].

individual_invite : [person -> event -> invite].
group_invite : [group -> event -> invite].


reminder : type.
none : reminder.
minutes_before : [int -> reminder].
hours_before : [int -> reminder].
days_before : [int -> reminder].

recurrence : type.
daily : recurrence.
weekly : recurrence.
monthly : recurrence.
yearly : recurrence.

availability :type.
busy : [person -> event -> availability].
free : [person -> event -> availability].
tentative : [person -> event -> availability].


category : type.
meeting : category.
conference : category.
social : category.
personal : category.

visibility : type.
public : visibility.
private : visibility.
confidential : visibility.

preferred_location : type.
preferred_time : type.
preferred_category : type.


user_preferences : type.

organizer : type.
participants : type.
attachments : type.

add_participant : [event -> person -> event -> action].
remove_participant : [event -> person -> event -> action].
add_attachment : [event -> attachments -> event -> action].
remove_attachment : [event -> attachments -> event -> action].

update_event : [event -> event -> event -> action].
change_visibility : [event -> visibility -> event -> action].
set_reminder : [event -> reminder -> event -> action].
mark_response : [person -> event -> availability -> action].
send_reminder_notification : [event -> action].
reschedule_event : [event -> duration -> event -> action].
cancel_event : [event -> action].
request_event_update : [person -> event -> action].
delegate_event : [event -> person -> action].
suggest_alternative_time : [person -> event -> duration -> action].
check_availability : [person -> event -> availability -> action].



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


