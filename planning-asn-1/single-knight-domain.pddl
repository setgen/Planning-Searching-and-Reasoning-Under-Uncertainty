(define (domain knight)
  (:requirements :strips)
  (:predicates
    (at ?p ?x ?y)
    (occupied ?x ?y)
    (delta1 ?a ?b)
    (delta2 ?a ?b)
  )

  (:action move
    :parameters (?p ?x1 ?y1 ?x2 ?y2)
    :precondition (and
      (at ?p ?x1 ?y1)
      (occupied ?x1 ?y1)
      (not (occupied ?x2 ?y2))
      (or
        (and (delta1 ?x1 ?x2) (delta2 ?y1 ?y2))
        (and (delta2 ?x1 ?x2) (delta1 ?y1 ?y2))
      )
    )
    :effect (and
      (not (at ?p ?x1 ?y1))
      (at ?p ?x2 ?y2)
      (not (occupied ?x1 ?y1))
      (occupied ?x2 ?y2)
    )
  )
)