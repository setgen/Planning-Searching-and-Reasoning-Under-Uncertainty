(define (domain king)
  (:requirements :strips)
  (:predicates
    (at ?p ?x ?y)
    (adjacent ?a ?b)
    (occupied ?x ?y)
  )

  (:action move
    :parameters (?p ?x1 ?y1 ?x2 ?y2)
    :precondition (and
      (at ?p ?x1 ?y1)
      (not (occupied ?x2 ?y2))
      (or (and (= ?x1 ?x2) (adjacent ?y1 ?y2))
          (and (= ?x1 ?x2) (adjacent ?y2 ?y1))
          (and (= ?y1 ?y2) (adjacent ?x1 ?x2))
          (and (= ?y1 ?y2) (adjacent ?x2 ?x1))
          (and (adjacent ?x1 ?x2) (adjacent ?y1 ?y2))
      )
    )
    :effect (and
      (not (at ?p ?x1 ?y1))
      (at ?p ?x2 ?y2)
      (occupied ?x2 ?y2)
      (not (occupied ?x1 ?x2))
    )
  )
)