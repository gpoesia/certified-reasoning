let b1 : person.
let b2 : person.
let b3 : person.

let h1 : group.

let f1 : event.
let f2 : event.
let f3 : event.

let org2 : (organizer f2 b2). 
let part2 : (participant f3 b3).
let rec3 : (monthly f1).
let rec4 : (yearly f2).
let prio2 : (high b1 f3).
let cat2 : (conference f1).

let inv3 : invite = (individual_invite b2 f2).
let inv4 : invite = (group_invite h1 f3).
let daxiom11 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (permissible (add_participant 'f b1))].
let daxiom12 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (obligatory (set_reminder (days_before 'b 'f)))].
let daxiom13 : [('f : event) -> ('h : group) -> (group_participant 'f 'h) -> (permissible (send_notification (group_invite 'h 'f)))]. 
let daxiom14 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (obligatory (check_availability 'b 'f))].
let daxiom15 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (accept (individual_invite 'b 'f)))].
let daxiom16 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (suggest_alternative_time 'b 'f))].
let daxiom17 : [('f : event) -> ('h : group) -> (group_participant 'f 'h) -> (obligatory (remove_participant 'f b3))].
let daxiom18 : [('f : event) -> (public 'f) -> (permissible (change_visibility 'f private))].
let daxiom19 : [('f : event) -> ('b : person) -> (low 'b 'f) -> (permissible (reschedule_event 'f yearly))].
let daxiom20 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let taxiom1 : [('f : event) -> (organizer 'f b1) -> (participant 'f b3)].
let taxiom2 : [('f : event) -> (group_participant 'f h1) -> (conference 'f)].
let taxiom3 : [('f : event) -> (monthly 'f) -> (busy b2 'f)].
let taxiom4 : [('f : event) -> (organizer 'f b2) -> (yearly 'f)].
let taxiom5 : [('f : event) -> (high b1 'f) -> (public 'f)].
Result:
(taxiom3 f1 rec3) : (busy b2 f1)
(daxiom15 f1 b2 r1) : (impermissible (accept (individual_invite b2 f1)))