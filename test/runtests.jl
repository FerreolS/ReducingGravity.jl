using Test

function run_tests()
    @testset "Mon ensemble de tests" begin
        @test 1 + 1 == 2
        @test "hello" * " world" == "hello world"
    end
end

run_tests()