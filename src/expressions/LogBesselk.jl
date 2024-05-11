export LogBesselk

import Bessels: besselkx

struct LogBesselk <: Expression
    ν::Number
end

function (f::LogBesselk)(x)
    return log(besselkx(f.ν, x)) - x
end