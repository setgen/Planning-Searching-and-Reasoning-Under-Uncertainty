# LLM Copilot did helped with some parts of this assignment see below 

import numpy as np

class StateGenerator:

    def __init__(self, nrows=8, ncols=7, npieces=10):
        """
        Initialize a generator for sampling valid states from
        an npieces dimensional state space.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.npieces = npieces
        self.rng = np.random.default_rng()

    def sample_state(self):
        """
        Samples a self.npieces length tuple.

        Output:
            Returns a state. A state is as 2-tuple (positions, dimensions), where
             -  Positions is represented as a list of position (c,r) tuples 
             -  Dimensions is a 2-tuple (self.nrows, self.ncols)

            For example, if the dimensions of the board are 2 rows, 3 columns, and the number of pieces
            is 4, then a valid return state would be ([(0, 0) , (1, 0), (2, 0), (1, 1)], (2,3))
        """
        ## Returns positions in decoded format. i.e. list of (c,r) i.e. (x,y)
        ## Without loss of generalization, we assume that positions[1:] are fixes; only
        ## positions[0] will be moved
        positions = self.rng.choice(self.nrows*self.ncols, size=self.npieces, replace=False)
        pos = list(self.decode(p) for p in positions)
        return pos, (self.nrows, self.ncols)

    def decode(self, position):
        r = position // self.ncols
        c = position - self.ncols * r
        return (c, r)

# LLM Copilot suggested or assisted with parts of this code block
# Helper function
def observation_distribution(true_pos, state):
    positions, (nrows, ncols) = state
    cx, cy = true_pos
    dist = np.zeros((nrows, ncols), dtype=float)

    occupied = set(positions[1:])

    center_prob = 0.6
    neighbor_prob = 0.1
    blocked_extra = 0.0

    cardinals = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dc, dr in cardinals:
        nx, ny = cx + dc, cy + dr
        if 0 <= nx < ncols and 0 <= ny < nrows and (nx, ny) not in occupied:
            dist[ny, nx] = neighbor_prob
        else:
            blocked_extra += neighbor_prob

    center_prob += blocked_extra
    if 0 <= cx < ncols and 0 <= cy < nrows:
        dist[cy, cx] += center_prob

    total = dist.sum()
    if total > 0:
        dist /= total
    return dist
# End of all suggested or assisted code in this block 

def sample_observation(state):
    """
    Given a state, sample an observation from it. Specifically, the positions[1:] locations are
    all known, while positions[0] should have a noisy observation applied.

    Input:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        A tuple (position, distribution) where:
         - Position is a sampled position which is a 2-tuple (c, r), which represents the sampled observation
         - Distribution is a 2D numpy array representing the observation distribution

    NOTE: the array representing the distribution should have a shape of (nrows, ncols)
    """
    positions, (nrows, ncols) = state
    true_pos = positions[0]

    dist = observation_distribution(true_pos, state)

    # LLM Copilot suggested or assisted with parts of this code block
    # Sample from the distribution
    flat = dist.ravel()
    rng = np.random.default_rng()
    idx = rng.choice(nrows * ncols, p=flat)
    r = idx // ncols
    c = idx % ncols
    return (c, r), dist
    # End of all suggested or assisted code in this block 

def sample_transition(state, action):
    """
    Given a state and an action, 
    returns:
         a resulting state, and a probability distribution represented by a 2D numpy array
    If a transition is invalid, returns None for the state, and a zero probability distribution
    NOTE: the array representing the distribution should have a shape of (nrows, ncols)

    Inputs:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        Action: a 2-tuple (dc, dr) representing the difference in positions of position[0] as a result of
                executing this transition.

    Outputs:
        A 2-tuple (new_position, transition_probabilities), where
            - new_position is:
                A 2-tuple (new_column, new_row) if the action is valid.
                None if the action is invalid.
            - transition_probabilities is a 2D numpy array with shape (nrows, ncols) that accurately reflects
                the probability of ending up at a certain position on the board given the action. 
    """
    positions, (nrows, ncols) = state
    (c, r) = positions[0]
    dc, dr = action

    # LLM Copilot suggested or assisted with parts of this code block
    new_c = c + dc
    new_r = r + dr

    trans = np.zeros((nrows, ncols), dtype=float)
    occupied = set(positions[1:])

    if not (0 <= new_c < ncols and 0 <= new_r < nrows):
        return None, trans
    if (new_c, new_r) in occupied:
        return None, trans

    trans[new_r, new_c] = 1.0
    return (new_c, new_r), trans
    # End of all suggested or assisted code in this block 
 
def initialize_belief(initial_state, style="uniform"):
    """
    Create an initial belief, based on the type of belief we want to start with

    Inputs:
        Initial_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        style: an element of the set {"uniform", "dirac"}

    Returns:
        an initial belief, represented by a 2D numpy array with shape (nrows, ncols)

    NOTE:
        The array representing the distribution should have a shape of (nrows, ncols).
        The occupied spaces (if any) should be zeroed out in the belief.
        We define two types of priors: a uniform prior (equal probability over all
        unoccupied spaces), and a dirac prior (which concentrates all the probability
        onto the actual position on the piece).
    
    """
    positions, (nrows, ncols) = initial_state
    belief = np.zeros((nrows, ncols), dtype=float)

    occupied = set(positions[1:])

    # LLM Copilot suggested or assisted with parts of this code block
    if style == "uniform":
        for r in range(nrows):
            for c in range(ncols):
                if (c, r) not in occupied:
                    belief[r, c] = 1.0
        total = belief.sum()
        if total > 0:
            belief /= total
        return belief

    if style == "dirac":
        c0, r0 = positions[0]
        if 0 <= c0 < ncols and 0 <= r0 < nrows and (c0, r0) not in occupied:
            belief[r0, c0] = 1.0
        total = belief.sum()
        if total > 0:
            belief /= total
        return belief
    # End of all suggested or assisted code in this block 

def belief_update(prior, observation, reference_state):
    """
    Given a prior an observation, compute the posterior belief

    Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        observation: a 2-tuple (col, row) representing the observation of a piece at a position
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    positions, (nrows, ncols) = reference_state
    obs_c, obs_r = observation

    posterior = np.zeros_like(prior, dtype=float)

    # LLM Copilot suggested or assisted with parts of this code block
    for r in range(nrows):
        for c in range(ncols):
            if prior[r, c] <= 0.0:
                continue
            true_pos = (c, r)
            dist = observation_distribution(true_pos, reference_state)
            likelihood = dist[obs_r, obs_c]
            if likelihood > 0.0:
                posterior[r, c] = prior[r, c] * likelihood

    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        posterior = prior.copy()
    return posterior
    # End of all suggested or assisted code in this block 

def belief_predict(prior, action, reference_state):
    """
    Given a prior, and an action, compute the posterior belief.

    Actions will be given in terms of dc, dr

   Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        action: a 2-tuple (dc, dr) as defined for action in sample_transition
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    positions, (nrows, ncols) = reference_state
    dc, dr = action

    posterior = np.zeros_like(prior, dtype=float)
    occupied = set(positions[1:])

    # LLM Copilot suggested or assisted with parts of this code block
    for r in range(nrows):
        for c in range(ncols):
            p = prior[r, c]
            if p <= 0.0:
                continue

            new_c = c + dc
            new_r = r + dr

            if not (0 <= new_c < ncols and 0 <= new_r < nrows):
                continue
            if (new_c, new_r) in occupied:
                continue

            posterior[new_r, new_c] += p

    total = posterior.sum()
    if total > 0:
        posterior /= total
    else:
        posterior = prior.copy()
    return posterior
    # End of all suggested or assisted code in this block 

if __name__ == "__main__":
    gen = StateGenerator()
    initial_state = gen.sample_state()
    obs, dist = sample_observation(initial_state)
    print(initial_state)
    print(obs)
    print(dist)
    b = initialize_belief(initial_state, style="uniform")
    print(b)
    b = belief_update(b, obs, initial_state)
    print(b)
    b = belief_predict(b, (1,0), initial_state)
    print(b)
