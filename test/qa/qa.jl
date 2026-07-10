using SciMLTesting, SpecializingFactorizations, Test

run_qa(
    SpecializingFactorizations;
    explicit_imports = true,
    ei_kwargs = (;
        # libblastrampoline's owner is libblastrampoline_jll; the package
        # legitimately ccalls it via LinearAlgebra.BLAS.
        all_explicit_imports_via_owners = (; ignore = (:libblastrampoline,)),
        # Non-public LinearAlgebra/LAPACK/BLAS internals this low-level
        # LAPACK-wrapping factorization package must use directly.
        all_explicit_imports_are_public = (;
            ignore = (
                Symbol("@blasfunc"), :BlasFloat, :BlasInt, :chkargsok, :hetrs!,
                :libblastrampoline, :potrf!, :potrs!, :sytrs!,
            ),
        ),
        # Non-public LinearAlgebra names accessed qualified (LinearAlgebra.QRPackedQ
        # for the QR factor view, LinearAlgebra.lutype for the lu promotion).
        all_qualified_accesses_are_public = (; ignore = (:QRPackedQ, :lutype)),
    )
)

@testset "public API appears in README" begin
    readme = read(joinpath(pkgdir(SpecializingFactorizations), "README.md"), String)
    missing = filter(name -> !occursin(String(name), readme), public_api_names(SpecializingFactorizations))
    @test isempty(missing)
end
