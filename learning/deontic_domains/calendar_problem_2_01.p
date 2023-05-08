let b1 : person.
let b2 : person.
let b3 : person.

let g2 : group.

let f1 : event.
let f2 : event.
let f3 : event.

let org2 : (organizer f2 b2). 
let part2 : (participant f3 b3).
let rec3 : (monthly f1).
let rec4 : (yearly f3).
let prio2 : (high b1 f1).
let cat2 : (conference f2).

let inv3 : invite = (individual_invite b2 f2).
let inv4 : invite = (group_invite g2 f3).
let daxiom11 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (permissible (send_notification (individual_invite 'b 'f)))].
let daxiom12 : [('f : event) -> ('g : group) -> (group_participant 'g 'f) -> (obligatory (accept (group_invite 'g 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (impermissible (cancel_event 'f))].
let daxiom14 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (set_reminder (days_before 'b 'f)))].
let daxiom16 : [('f : event) -> (public 'f) -> (permissible (change_visibility 'f private))].
let daxiom17 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (suggest_alternative_time 'b 'f))].
let daxiom21 : [('f : event) -> (yearly 'f) -> (impermissible (change_visibility 'f private))]
let daxiom18 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (obligatory (decline (individual_invite 'b 'f)))].
let daxiom19 : [('f : event) -> (short 'f) -> (permissible (update_event 'f social))].
let daxiom20 : [('f : event) -> ('g : group) -> (group_participant 'g 'f) -> (obligatory (check_availability b1 'f))].
let taxiom1 : [('f : event) -> (organizer 'f b1) -> (participant 'f b2)].
let taxiom2 : [('f : event) -> (conference 'f) -> (public 'f)].
let taxiom3 : [('f : event) -> (short 'f) -> (social 'f)].
Result:
(taxiom2 f2 cat2) : (public f2)
(daxiom16 f2 r1) : (permissible (change_visibility f2 private))