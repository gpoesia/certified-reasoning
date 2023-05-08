let b1 : person.
let b2 : person.
let b3 : person.
let h1 : group.
let f1 : event.
let f2 : event.
let f3 : event.

let part2 : (participant f3 b3).
let rec3 : (monthly f1).
let rec4 : (yearly f2).
let cat2 : (conference f1).
let dur1 : (short f1).
let dur2 : (long f3).
let rem1 : reminder = (hours_before b1 f1).
let rem2 : reminder = (days_before b2 f2).
let avail1 : (busy b1 f1).
let avail2 : (free b2 f3).
let vis1 : (public f3).
let vis2 : (private f1).

let daxiom11 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (add_participant 'f 'b))].
let daxiom12 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (add_participant 'f 'b))].
let daxiom13 : [('f : event) -> ('h : group) -> (long 'f) -> (obligatory (add_participant 'f 'h))].
let daxiom14 : [('f : event) -> ('b : person) -> (high 'b 'f) -> (obligatory (set_reminder (days_before 'b 'f)))].
let daxiom15 : [('f : event) -> (conference 'f) -> (permissible (update_event 'f public))].
let daxiom21 : [('f : event) -> ('b : person) -> (low 'b 'f) -> (impermissible (request_event_update 'b 'f))].
let daxiom16 : [('f : event) -> ('b : person) -> (low 'b 'f) -> (permissible (cancel_event 'f))].
let daxiom17 : [('f : event) -> ('b : person) -> (short 'f) -> (impermissible (reschedule_event 'f yearly))].
let daxiom18 : [('f : event) -> (private 'f) -> (obligatory (remove_participant 'f b3))].
let daxiom19 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (request_event_update 'b 'f))].
let daxiom20 : [('f : event) -> (organizer 'f b1) -> (obligatory (change_visibility 'f confidential))].

let taxiom1 : [('f : event) -> (monthly 'f) -> (short 'f)].
let taxiom2 : [('f : event) -> (yearly 'f) -> (long 'f)].
let taxiom3 : [('b : person) -> ('f : event) -> (long 'f) -> (high 'b 'f)].
let taxiom4 : [('b : person) -> ('f : event) -> (short 'f) -> (free 'b 'f)].
let taxiom5 : [('f : event) -> (long 'f) -> (organizer 'f b1)].
let taxiom6 : [('b : person) -> ('f : event) -> (free 'b 'f) -> (high 'b 'f)].
let taxiom7 : [('b : person) -> ('f : event) -> (days_before 'b 'f) -> (low 'b 'f)].
let taxiom8 : [('f : event) -> (private 'f) -> (meeting 'f)].
let taxiom9 : [('f : event) -> (public 'f) -> (social 'f)].
let taxiom10 : [('b : person) -> ('f : event) -> (organizer 'f 'b) -> (participant 'f 'b)].

Result:
(taxiom2 f2 rec4) : (long f2)
(taxiom5 f2 r1) : (organizer f2 b1)
(taxiom10 b1 f2 r2) : (participant f2 b1)
(daxiom19 f2 b1 r3) : (permissible (request_event_update b1 f2))