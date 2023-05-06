action : type.

/* base deontic types */
obligatory : [action -> prop].
permissible : [action -> prop].
impermissible : [action -> prop].

/* base axioms */
ob_perm : [(obligatory 'a) -> (permissible 'a)].
im_perm : [(impermissible 'a) -> (not (permissible 'a))].

event : type.
entity : type.
person : entity.
group : entity.


invite : type.
individual_invite : [person -> event -> invite].
group_invite : [group -> event -> invite].
send_notification : [invite -> action].
accept : [invite -> action].
decline : [invite -> action].
cancel_event : [event -> action].

reminder : type.
none : [person -> event -> reminder].
minutes_before : [person -> event -> reminder].
hours_before : [person -> event -> reminder].
days_before : [person -> event -> reminder].
set_reminder : [reminder -> action].


duration : type.
short : [event -> prop].
long : [event -> prop].

priority: type.
high : [person -> event -> prop].
low : [person -> event -> prop].


recurrence : type.
daily : [event -> prop].
weekly : [event -> prop].
monthly : [event -> prop].
yearly : [event -> prop].

availability :type.
busy : [person -> event -> prop].
free : [person -> event -> prop].
tentative : [person -> event -> prop].
check_availability : [person -> event -> action].

category : type.
meeting : [event -> prop].
conference : [event -> prop].
social : [event -> prop].
personal : [event -> prop].

visibility : type.
public : [event -> prop].
private : [event -> prop].
confidential : [event -> prop].
change_visibility : [event -> [event -> prop] -> action].


organizer : [event -> person -> prop].
participant : [event -> person -> prop].
group_participant : [event -> group -> prop].

add_participant : [event -> entity -> action].
remove_participant : [event -> entity -> action].
delegate_event : [event -> person -> action].

update_event : [event -> [event -> prop] -> action].
reschedule_event : [event -> [event -> prop] -> action].
request_event_update : [person -> event -> action].
suggest_alternative_time : [person -> event -> action].
