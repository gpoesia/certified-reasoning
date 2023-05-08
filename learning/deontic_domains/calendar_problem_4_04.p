let b1 : person.
let b2 : person.
let b3 : person.
let g2 : group.
let f1 : event.
let f2 : event.
let f3 : event.

let org2 : (organizer f1 b1).
let part2 : (participant f1 b3).

let vis1 : (public f1).
let vis2 : (confidential f3).
let dur1 : (short f2).
let cat2 : (conference f1).
let cat3 : (social f3).
let rec3 : (monthly f2).

let reminder1 : reminder = (hours_before b1 f1).
let reminder2 : reminder = (days_before b2 f2).
let inv3 : invite = (individual_invite b1 f1).

let daxiom11 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (accept (individual_invite 'b 'f)))].
let daxiom12 : [('f : event) -> ('g : group) -> (group_participant 'g 'f) -> (permissible (send_notification (group_invite 'g 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (check_availability 'b 'f))].
let daxiom14 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let daxiom15 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (obligatory (update_event 'f social))].
let daxiom16 : [('f : event) -> ('b : person) -> (long 'f) -> (permissible (set_reminder (days_before 'b 'f)))].
let daxiom18 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (reschedule_event 'f daily))].
let daxiom19 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (suggest_alternative_time 'b 'f))].
let daxiom20 : [('f : event) -> (long 'f) -> (permissible (change_visibility 'f confidential))].

let taxiom1 : [('f : event) -> (social 'f) -> (public 'f)].
let taxiom2 : [('b : person) -> ('f : event) -> (busy 'b 'f) -> (low 'b 'f)].
let taxiom3 : [('f : event) -> (public 'f) -> (long 'f)].
let taxiom4 : [('f : event) -> (long 'f) -> (free b3 'f)].
let taxiom5 : [('b : person) -> ('f : event) -> (organizer 'b 'f) -> (confidential 'f)].
let taxiom6 : [('b : person) -> ('f : event) -> (free 'b 'f) -> (low 'b 'f)].
let taxiom7 : [('f : event) -> (short 'f) -> (conference 'f)].
let taxiom8 : [('b : person) -> ('f : event) -> (participant 'b 'f) -> (daily 'f)].
let taxiom10 : [('b : person) -> ('f : event) -> (busy 'b 'f) -> (yearly 'f)].

Result:
(taxiom1 f3 cat3) : (public f3)
(taxiom3 f3 r1) : (long f3)
(taxiom4 f3 r2) : (free b3 f3)
(daxiom19 f3 b3 r3) : (obligatory (suggest_alternative_time b3 f3))