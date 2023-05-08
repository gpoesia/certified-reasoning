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
let cat2 : (conference f1).
let dur1 : (short f1).
let dur2 : (long f3).
let rem1 : reminder = (hours_before b1 f1).
let rem2 : reminder = (days_before b2 f2).
let avail1 : (busy b1 f1).
let avail2 : (free b2 f3).
let vis1 : (public f3).
let vis2 : (private f1).

let daxiom11 : [('f : event) -> (public 'f) -> (permissible (add_participant 'f b1))].
let daxiom12 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (accept (individual_invite 'b 'f)))].
let daxiom13 : [('f : event) -> ('b : person) -> (organizer 'f 'b) -> (obligatory (set_reminder (days_before 'b 'f)))].
let daxiom14 : [('f : event) -> ('b : person) -> (participant 'f 'b) -> (permissible (delegate_event 'f 'b))].
let daxiom15 : [('f : event) -> ('h : group) -> (group_participant 'f 'h) -> (obligatory (update_event 'f social))].
let daxiom16 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (permissible (reschedule_event 'f yearly))].
let daxiom17 : [('f : event) -> ('b : person) -> (busy 'b 'f) -> (impermissible (suggest_alternative_time 'b 'f))].
let daxiom18 : [('f : event) -> ('b : person) -> (free 'b 'f) -> (obligatory (check_availability 'b 'f))].
let daxiom19 : [('f : event) -> ('b : person) -> (short 'f) -> (permissible (change_visibility 'f private))].
let daxiom20 : [('f : event) -> ('b : person) -> (long 'f) -> (obligatory (set_reminder (minutes_before 'b 'f)))].
Result:
(daxiom18 f3 b2 avail2) : (obligatory (check_availability b2 f3))