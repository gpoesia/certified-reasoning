let b1 : person.
let b2 : person.
let b3 : person.
let g2 : group.
let e4 : event.
let e5 : event.
let e6 : event.

let org2 : (organizer e4 b1).
let part2 : (participant e5 b3).
let group_part1 : (group_participant e6 g2).
let dur1 : (short e4).
let dur2 : (long e5).
let avail1 : (busy b2 e6).
let cat2 : (meeting e4).
let cat3 : (social e5).
let rem1 : reminder = (hours_before b1 e4).

let daxiom11 : [('e : event) -> ('a : person) -> (organizer 'e 'a) -> (obligatory (add_participant 'e 'a))].
let daxiom12 : [('e : event) -> ('a : person) -> (free 'a 'e) -> (permissible (accept(individual_invite 'a 'e)))].
let daxiom13 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (permissible (delegate_event 'e 'g))].
let daxiom14 : [('e : event) -> ('a : person) -> (busy 'a 'e) -> (impermissible (set_reminder (minutes_before 'a 'e)))].
let daxiom15 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (remove_participant 'e 'a))].
let daxiom16 : [('e : event) -> ('a : person) -> (short 'e) -> (permissible (update_event 'e social))].
let daxiom17 : [('e : event) -> ('a : person) -> (tentative 'a 'e) -> (obligatory (check_availability 'a 'e))].
let daxiom18 : [('e : event) -> ('a : person) -> (high 'a 'e) -> (obligatory (set_reminder (days_before 'a 'e)))].
let daxiom19 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (suggest_alternative_time 'a 'e))].
let daxiom20 : [('e : event) -> (public 'e) -> (obligatory (add_participant b1 'e))].

let taxiom1 : [('e : event) -> (meeting 'e) -> (public 'e)].
let taxiom2 : [('e : event) -> (public 'e) -> (short 'e)].
let taxiom3 : [('e : event) -> ('a : person) -> (organizer 'e 'a) -> (high 'e 'a)].
let taxiom4 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (available 'a 'e)].
let taxiom5 : [('e : event) -> (short 'e) -> (participant 'e b2)].
let taxiom6 : [('e : event) -> (daily 'e) -> (short 'e)].
let taxiom7 : [('a : person) -> ('e : event) -> (high 'a 'e) -> (busy 'a 'e)].
let taxiom8 : [('a : person) -> ('e : event) -> (low 'a 'e) -> (free 'a 'e)].
let taxiom9 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (public 'e)].

Result:
(taxiom9 e6 g2 group_part1) : (public e6)
(taxiom2 e6 r1) : (short e6)
(taxiom5 e6 r2) : (participant e6 b2)
(daxiom19 e6 b2 r3) : (permissible (suggest_alternative_time b2 e6))