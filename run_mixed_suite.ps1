param(
    [Parameter(Mandatory = $true)]
    [string]$RunId
)

$ErrorActionPreference = "Continue"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$benchmark = Join-Path $PSScriptRoot "mixed_precision_benchmark.py"
$runFolder = Join-Path $PSScriptRoot "benchmark_logs\$RunId"
New-Item -ItemType Directory -Force -Path $runFolder | Out-Null

$configs = @(
    @{Case=1; Batch=64; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=2; Batch=1; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=3; Batch=4; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=4; Batch=16; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=5; Batch=128; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=6; Batch=10000; Qkv=128; Heads=4; Seq=128; Layers=4; Ffn=128},
    @{Case=7; Batch=64; Qkv=32; Heads=4; Seq=128; Layers=4; Ffn=32},
    @{Case=8; Batch=64; Qkv=1024; Heads=4; Seq=128; Layers=4; Ffn=1024},
    @{Case=9; Batch=64; Qkv=128; Heads=1; Seq=128; Layers=4; Ffn=128},
    @{Case=10; Batch=64; Qkv=128; Heads=2; Seq=128; Layers=4; Ffn=128},
    @{Case=11; Batch=64; Qkv=128; Heads=16; Seq=128; Layers=4; Ffn=128},
    @{Case=12; Batch=64; Qkv=128; Heads=4; Seq=32; Layers=4; Ffn=128},
    @{Case=13; Batch=64; Qkv=128; Heads=4; Seq=1024; Layers=4; Ffn=128},
    @{Case=14; Batch=32; Qkv=1024; Heads=16; Seq=100000; Layers=2; Ffn=1024}
)

$results = foreach ($c in $configs) {
    Write-Host "CASE $($c.Case) START"
    $output = & $python $benchmark `
        --batch-size $c.Batch `
        --d-model $c.Qkv `
        --heads $c.Heads `
        --seq-len $c.Seq `
        --ffn-dim $c.Ffn `
        --layers $c.Layers `
        --causal `
        --accuracy-trials 1 `
        --warmup 3 `
        --repeats 10 `
        --benchmark-rounds 1 2>&1
    $exitCode = $LASTEXITCODE
    $text = $output -join "`n"
    $output | Out-File (Join-Path $runFolder "case_$($c.Case).txt")

    $status = if ($exitCode -eq 0) { "OK" } elseif ($text -match "out of memory|OutOfMemoryError") { "OOM" } else { "ERROR" }
    $accuracy = if ($text -match 'summary: (PASS|FAIL)') { $Matches[1] } else { "N/A" }
    $maxAbs = if ($text -match 'summary: (?:PASS|FAIL) \| max_abs=([^|]+)') { $Matches[1].Trim() } else { "N/A" }
    $failed = if ($text -match 'summary: (?:PASS|FAIL).*failed=([^\r\n]+)') { $Matches[1].Trim() } else { "N/A" }
    $baseline = if ($text -match 'baseline : median=([0-9.]+) ms') { $Matches[1] } else { "N/A" }
    $candidate = if ($text -match 'optimized: median=([0-9.]+) ms') { $Matches[1] } else { "N/A" }
    $speedup = if ($text -match 'speedup  : ([0-9.]+)x') { $Matches[1] } else { "N/A" }
    Write-Host "CASE $($c.Case) $status accuracy=$accuracy candidate_ms=$candidate speedup=$speedup"

    [pscustomobject]@{
        Case=$c.Case; Batch=$c.Batch; DModel=$c.Qkv; Heads=$c.Heads;
        SeqLen=$c.Seq; Layers=$c.Layers; FfnDim=$c.Ffn; Status=$status;
        Accuracy=$accuracy; MaxAbs=$maxAbs; Failed=$failed;
        StrictFp32BaselineMs=$baseline; MixedCandidateMs=$candidate; Speedup=$speedup
    }
}

$results | Export-Csv (Join-Path $runFolder "results.csv") -NoTypeInformation
$results | Format-Table -AutoSize
