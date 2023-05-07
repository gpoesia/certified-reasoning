let b1 : person.
let b2 : person.
let b3 : person.
let g2 : group.
let f1 : event.
let f2 : event.

let org2 : (organizer f1 b1).
let part2 : (participant f1 b2).
let grp1 : (group_participant f1 g2).
let short1 : (short f2).
let cat1 : (social f1).
let reminder1 : reminder = (hours_before b1 f1).
let visibility1 : (public f1).

let daxiom11 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (obligatory (add_participant 'f 'b))].
let daxiom12 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (accept (individual_invite 'b 'f)))]. 
let daxiom13 : [('f : event) -> (short 'f) -> (impermissible (cancel_event 'f))].
let daxiom14 : [('f : event) -> ('g : group) -> (group_participant 'f 'g) -> (obligatory (add_participant 'f 'g))].
let daxiom15 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (set_reminder (hours_before 'b 'f)))]. 
let daxiom16 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (reschedule_event 'f daily))].
let daxiom17 : [('f : event) -> (public 'f) -> (obligatory (check_availability b1 'f))].
let daxiom18 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let daxiom19 : [('f : event) -> ('b : person) -> (tentative 'b 'f) -> (permissible (suggest_alternative_time 'b 'f))].
let daxiom20 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (obligatory (set_reminder (days_before 'b 'f)))].

let taxiom11 : [('f : event) -> (short 'f) -> (meeting 'f)].
let taxiom12 : [('f : event) -> ('b : person) -> (meeting 'f) -> (tentative 'b 'f)].
let taxiom13 : [('f : event) -> (tentative b1 'f) -> (participant 'f b1)].
let taxiom14 : [('b : person) -> ('f : event) -> (free 'b 'f) -> (low 'f 'b)].
let taxiom15 : [('f : event) -> (public 'f) -> (social 'f)].
let taxiom16 : [('f : event) -> (private 'f) -> (personal 'f)].
let taxiom17 : [('b : person) -> ('f : event) -> (participant 'f 'b) -> (organizer 'f 'b) -> (not (group_participant 'f 'b))].
let taxiom18 : [('f : event) -> (daily 'f) -> (not (weekly 'f))].
let taxiom19 : [('f : event) -> (weekly 'f) -> (not (monthly 'f))].
let taxiom20 : [('b : person) -> ('f : event) -> (set_reminder (hours_before 'b 'f)) -> (not (set_reminder (days_before 'b 'f)))].

Result:
(taxiom11 f2 short1) : (meeting f2)
(taxiom12 f2 b2 r1) : (tentative b2 f2)
(daxiom19 f2 b2 r2) : (permissible (suggest_alternative_time b2 f2))