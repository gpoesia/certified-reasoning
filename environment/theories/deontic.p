action : type.
event : type.
person : type.
group : type.
message : type.
invite : type.

obligatory : [action -> prop].
permissible : [action -> prop].
optional: [action -> prop].
omissible : [action -> prop].
impermissible : [action -> prop].
notoptional: [action -> prop].

duration : type.
short : duration.
long : duration.

location : type.

priority : type.
high : priority.
low : priority.

send_notification : [invite -> action].

accept : [invite -> action].
decline : [invite -> action].

individual_invite : [person -> event -> invite].
group_invite : [group -> event -> invite].


recurrence : type.
daily : recurrence.
weekly : recurrence.
monthly : recurrence.
yearly : recurrence.

availability : type.
busy : availability.
free : availability.
tentative : availability.


category : type.
meeting : category.
conference : category.
social : category.
personal : category.

visibility : type.
public : visibility.
private : visibility.
confidential : visibility.

preferred_location : type.
preferred_time : type.
preferred_category : type.


user_preferences : type.

organizer : type.
participants : type.
attachments : type.

add_participant : [event -> person -> event -> action].
remove_participant : [event -> person -> event -> action].
add_attachment : [event -> attachments -> event -> action].
remove_attachment : [event -> attachments -> event -> action].

update_event : [event -> event -> event -> action].
change_visibility : [event -> visibility -> event -> action].
set_reminder : [event -> reminder -> event -> action].
mark_response : [person -> event -> availability -> action].
send_reminder_notification : [event -> action].
reschedule_event : [event -> duration -> event -> action].
cancel_event : [event -> action].
request_event_update : [person -> event -> action].
delegate_event : [event -> person -> action].
suggest_alternative_time : [person -> event -> duration -> action].
check_availability : [person -> event -> availability -> action].


/*  */

verify basic_invite {
       let a : person.
       let b : person.
       let g1 : group.
       let e : event.

       let axiom1 : [('e : event) ->
               (permissible (accept (individual_invite a 'e)))].
       let axiom2 : [('e : event) -> ('g : group) ->
               (permissible (decline (group_invite 'g 'e)))].

       let axiom3 : [('i : invite) ->
                     (permissible (accept 'i)) ->
                     (permissible (send_notification 'i))].

       let i1 : invite = (individual_invite a e).
       let i2 : invite = (group_invite g1 e).

       show (permissible (accept i1)) by axiom1.
       show (permissible (send_notification i1)) by axiom3.
       show (permissible (decline i2)) by axiom2.

}

