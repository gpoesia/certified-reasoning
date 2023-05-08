let b1 : person.
let b2 : person.
let b3 : person.

let g2 : group.

let f1 : event.
let f2 : event.
let f3 : event.

let org2 : (organizer f2 b2). 
let part2 : (participant f3 b1).
let rec3 : (monthly f1).
let rec4 : (yearly f3).
let prio2 : (high b2 f1).
let cat2 : (conference f2).

let inv3 : invite = (individual_invite b3 f3).
let inv4 : invite = (group_invite g2 f1).
let daxiom11 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (obligatory (send_notification (individual_invite 'b 'f)))].
let daxiom12 : [('f : event) -> ('g : group) -> (group_participant 'g 'f) -> (permissible (accept (group_invite 'g 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (set_reminder (days_before 'b 'f)))].
let daxiom14 : [('f : event) -> (public 'f) -> (obligatory (check_availability b3 'f))].
let daxiom15 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (suggest_alternative_time 'b 'f))].
let daxiom16 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (reschedule_event 'f yearly))].
let daxiom17 : [('f : event) -> ('b : person) -> (tentative 'b 'f) -> (permissible (delegate_event 'f 'b))].
let daxiom18 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (obligatory (update_event 'f conference))].
let daxiom19 : [('f : event) -> ('b : person) -> (low 'b 'f) -> (permissible (change_visibility 'f public))].
let daxiom20 : [('f : event) -> ('g : group) -> (group_participant 'f 'g) -> (impermissible (remove_participant 'f 'g))].
let taxiom1 : [('f : event) -> (organizer 'f b2) -> (participant 'f b1)].
let taxiom2 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (long 'f)].
let taxiom4 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (social 'f)].
let taxiom5 : [('f : event) -> ('b : person) -> (tentative 'b 'f) -> (personal 'f)].
Result:
(taxiom1 f2 org2) : (participant f2 b1)
(daxiom13 f2 b1 r1) : (permissible (set_reminder (days_before b1 f2)))