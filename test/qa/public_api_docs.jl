using SpecializingFactorizations
using Test

@testset "public API documentation" begin
    public_names = filter(
        !=(:SpecializingFactorizations),
        names(SpecializingFactorizations; all = false, imported = false),
    )
    expected_names = Set(
        [
            :BANDED,
            :DIAGONAL,
            :DetectionResult,
            :GENERAL,
            :HERMITIAN_INDEFINITE,
            :LOWER_BIDIAGONAL,
            :LOWER_TRIANGULAR,
            :MatrixForm,
            :QRStatus,
            :QR_DEFICIENT,
            :QR_FULLRANK,
            :QR_UNFACTORED,
            :SYMMETRIC_INDEFINITE,
            :SYMMETRIC_POSITIVE_DEFINITE,
            :SpecializedLU,
            :SpecializedQR,
            :TRIDIAGONAL,
            :UPPER_BIDIAGONAL,
            :UPPER_TRIANGULAR,
            :detect_form,
            :isfactored,
            :issuccess,
            :matrixform,
            :reserve!,
            :specializinglu,
            :specializinglu!,
            :specializingqr,
            :specializingqr!,
            :structuralform,
        ]
    )
    @test Set(public_names) == expected_names

    for name in public_names
        binding = Docs.Binding(SpecializingFactorizations, name)
        @test Docs.hasdoc(binding)
    end

    readme = read(joinpath(pkgdir(SpecializingFactorizations), "README.md"), String)
    for name in public_names
        @test occursin(string(name), readme)
    end
end
