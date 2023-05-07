let a1 : person.
let a2 : person.
let a3 : person.
let g1 : group.

let e1 : event.
let e2 : event.
let e3 : event.

let org1 : (organizer e1 a1). 
let part1 : (participant e1 a2).
let rec1 : (daily e3).
let dur1 : (short e2).
let cat1 : (meeting e3).

let inv1 : invite = (individual_invite a1 e1).
let inv2 : invite = (group_invite g1 e2).

let daxiom1 : [('e : event) -> ('a : person) -> (free 'e 'a) -> (permissible (send_notification 'b 'f))].
let daxiom2 : [('e : event) -> ('g : group) -> (group_participant 'g 'e) -> (permissible (accept (group_invite 'g 'e)))].
let daxiom3 : [('e : event) -> ('a : person) -> (busy 'a 'e) -> (impermissible (reschedule_event 'e daily))].
let daxiom4 : [('e : event) -> ('a : person) -> (high 'a 'e) -> (obligatory (set_reminder (hours_before 'a 'e)))].
let daxiom5 : [('e : event) -> ('a : person) -> (participant 'e 'a) -> (permissible (delegate_event 'e 'a))].
let daxiom6 : [('e : event) -> ('a : person) -> (long 'e) -> (permissible (update_event 'e conference))].
let daxiom7 : [('e : event) -> ('g : group) -> (group_participant 'e 'g) -> (impermissible (remove_participant 'e 'g))].
let daxiom8 : [('e : event) -> ('a : person) -> (free 'a 'e) -> (obligatory (accept (individual_invite 'a 'e)))].
let daxiom9 : [('e : event) -> ('a : person) -> (tentative 'a 'e) -> (permissible (suggest_alternative_time 'a 'e))].
let daxiom10 : [('e : event) -> (private 'e) -> (obligatory (check_availability a1 'e))].

let taxiom1 : [('e : event) -> ((individual_invite a1 'e): invite) -> (short 'e)].
let taxiom2 : [('e : event) -> (daily 'e) -> (long 'e)].
let taxiom3 : [('e: event) -> (long 'e) -> (private 'e)].
let taxiom4 : [('e : event) -> (private 'e) -> (participant 'e a1)].
let taxiom5 : [('p : person) -> ('e : event) -> (high 'e 'p) -> (free 'e 'p)].
let taxiom6: [('e : event) -> (weekly 'e) -> (high a2 'e)].
let taxiom7 : [('e : event) -> (meeting 'e) -> (organizer 'e a3)].
let taxiom8 : [('p : person) -> (organizer e1 'p) -> (high 'p e1)].
let taxiom9 : [('e : event) -> (organizer 'e a1) -> (confidential 'e)].
let taxiom10 : [('e : event) -> (participant 'e a2) -> (days_before a2 'e)].

Result:
(taxiom2 e3 rec1) : (long e3)
(taxiom3 e3 r1) : (private e3)
(taxiom4 e3 r2) : (participant e3 a1)
(daxiom5 e3 a1 r3) : (permissible (delegate_event e3 a1))