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
let rec4 : (yearly f3).
let prio2 : (high b1 f1).
let cat2 : (conference f2).

let inv3 : invite = (individual_invite b3 f3).
let inv4 : invite = (group_invite h1 f1).
let daxiom12 : [('f : event) -> ('h : group) -> (group_participant 'h 'f) -> (permissible (accept (group_invite 'h 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (reschedule_event 'f monthly))].
let daxiom14 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (obligatory (set_reminder (days_before 'b 'f)))].
let daxiom15 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let daxiom16 : [('f : event) -> (conference 'f) -> (permissible (update_event 'f public))].
let daxiom17 : [('f : event) -> ('h : group) -> (group_participant 'f 'h) -> (impermissible (remove_participant 'f 'h))].
let daxiom18 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (accept (individual_invite 'b 'f)))].
let daxiom19 : [('f : event) -> ('b : person) -> (tentative 'b 'f) -> (permissible (suggest_alternative_time 'b 'f))].
let daxiom20 : [('f : event) -> (private 'f) -> (obligatory (check_availability b2 'f))].
let taxiom1 : [('f : event) -> (organizer 'f b2) -> (participant 'f b3)].
let taxiom2 : [('f : event) -> (yearly 'f) -> (high b1 'f) -> (conference 'f)].
let taxiom3 : [('f : event) -> (monthly 'f) -> (busy b1 'f) -> (meeting 'f)].
let taxiom4 : [('f : event) -> (participant 'f b3) -> (free b3 'f) -> (delegate_event 'f b3)].
let taxiom5 : [('f : event) -> (group_participant 'f h1) -> (organizer 'f b2) -> (public 'f)].
Result:
(taxiom1 f2 org2) : (participant f2 b3)
(daxiom15 f2 b3 r1) : (permissible (delegate_event f2 b3))