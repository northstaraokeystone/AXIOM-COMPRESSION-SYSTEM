"""Gravity constants for celestial bodies.

Values in Earth g units (1g = 9.81 m/s^2).
"""

# Inner planets
GRAVITY_EARTH = 1.0
GRAVITY_MARS = 0.38
GRAVITY_VENUS = 0.9
GRAVITY_MERCURY = 0.38

# Earth's moon
GRAVITY_MOON = 0.166

# Saturnian system
GRAVITY_TITAN = 0.14

# Jovian system
GRAVITY_EUROPA = 0.134
GRAVITY_GANYMEDE = 0.146
GRAVITY_CALLISTO = 0.126
GRAVITY_IO = 0.183

# Outer planets (surface gravity at 1 bar level)
GRAVITY_JUPITER = 2.528
GRAVITY_SATURN = 1.065
GRAVITY_URANUS = 0.886
GRAVITY_NEPTUNE = 1.14

# Dwarf planets
GRAVITY_PLUTO = 0.063
GRAVITY_CERES = 0.029
GRAVITY_ERIS = 0.082

# Complete planet gravity map
PLANET_GRAVITY_MAP = {
    # Inner planets
    "earth": GRAVITY_EARTH,
    "mars": GRAVITY_MARS,
    "venus": GRAVITY_VENUS,
    "mercury": GRAVITY_MERCURY,
    # Earth's moon
    "moon": GRAVITY_MOON,
    "luna": GRAVITY_MOON,
    # Saturnian system
    "titan": GRAVITY_TITAN,
    # Jovian system
    "europa": GRAVITY_EUROPA,
    "ganymede": GRAVITY_GANYMEDE,
    "callisto": GRAVITY_CALLISTO,
    "io": GRAVITY_IO,
    # Outer planets
    "jupiter": GRAVITY_JUPITER,
    "saturn": GRAVITY_SATURN,
    "uranus": GRAVITY_URANUS,
    "neptune": GRAVITY_NEPTUNE,
    # Dwarf planets
    "pluto": GRAVITY_PLUTO,
    "ceres": GRAVITY_CERES,
    "eris": GRAVITY_ERIS,
    # Jovian system alias
    "jovian_system": GRAVITY_EUROPA,  # Use Europa as reference
}

# Gravity categories for operational planning
GRAVITY_CATEGORIES = {
    "micro": (0.0, 0.1),  # Asteroids, small moons
    "low": (0.1, 0.3),  # Jovian moons, Pluto
    "medium": (0.3, 0.6),  # Mars, Mercury
    "high": (0.6, 1.0),  # Venus
    "earth_like": (0.9, 1.1),  # Earth
    "super": (1.1, 3.0),  # Gas giants
}


def get_gravity_category(gravity_g: float) -> str:
    """Get category for a gravity value.

    Args:
        gravity_g: Gravity in Earth g units.

    Returns:
        str: Gravity category name.
    """
    for category, (low, high) in GRAVITY_CATEGORIES.items():
        if low <= gravity_g < high:
            return category
    return "unknown"
