$runID = Read-Host "Enter a run ID for this sweep (e.g. baseline, sdpa-v1, fused-attn-v2)"
if ([string]::IsNullOrWhiteSpace($runID)) {
    $runID = "run_" + (Get-Date -Format "yyyyMMdd_HHmmss")
    Write-Host "No run ID entered, defaulting to: $runID"
}
 
$invalidChars = [System.IO.Path]::GetInvalidFileNameChars()
foreach ($ch in $invalidChars) {
    $runID = $runID.Replace([string]$ch, "_")
}
 
$runFolder = "logs\$runID"
New-Item -ItemType Directory -Force -Path $runFolder | Out-Null
Write-Host "Output folder: $runFolder"
Write-Host ""
 
$configs = @(
    @{n=1;  batch=64;    qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=2;  batch=1;     qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=3;  batch=4;     qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=4;  batch=16;    qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=5;  batch=128;   qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=6;  batch=10000; qkv=128;  heads=4;  seq=128;    layers=4; ffn=128}
    @{n=7;  batch=64;    qkv=32;   heads=4;  seq=128;    layers=4; ffn=32}
    @{n=8;  batch=64;    qkv=1024; heads=4;  seq=128;    layers=4; ffn=1024}
    @{n=9;  batch=64;    qkv=128;  heads=1;  seq=128;    layers=4; ffn=128}
    @{n=10; batch=64;    qkv=128;  heads=2;  seq=128;    layers=4; ffn=128}
    @{n=11; batch=64;    qkv=128;  heads=16; seq=128;    layers=4; ffn=128}
    @{n=12; batch=64;    qkv=128;  heads=4;  seq=32;     layers=4; ffn=128}
    @{n=13; batch=64;    qkv=128;  heads=4;  seq=1024;   layers=4; ffn=128}
    @{n=14; batch=32;    qkv=1024; heads=16; seq=100000; layers=2; ffn=1024}
)
 
$results = @()
 
foreach ($c in $configs) {
    Write-Host "=== Case $($c.n): batch=$($c.batch) qkv=$($c.qkv) heads=$($c.heads) seq=$($c.seq) layers=$($c.layers) ==="
 
    $output = python .\torch_transformer_benchmark.py --batch-size $c.batch --d-model $c.qkv --heads $c.heads --seq-len $c.seq --ffn-dim $c.ffn --layers $c.layers --causal --rtol 0.02 --atol 0.002 2>&1
 
    $exitCode = $LASTEXITCODE
    $accuracyLine = $output | Select-String "summary:"
    $speedupLine = $output | Select-String "speedup"
 
    $status = "ERROR"
    if ($exitCode -eq 0) {
        $status = "OK"
    } elseif ($output -match "out of memory") {
        $status = "OOM"
    } elseif ($output -match "OutOfMemoryError") {
        $status = "OOM"
    }
 
    $resultObj = New-Object PSObject
    $resultObj | Add-Member -MemberType NoteProperty -Name "Case" -Value $c.n
    $resultObj | Add-Member -MemberType NoteProperty -Name "Status" -Value $status
    $resultObj | Add-Member -MemberType NoteProperty -Name "ExitCode" -Value $exitCode
    $resultObj | Add-Member -MemberType NoteProperty -Name "Accuracy" -Value $(if ($accuracyLine) { $accuracyLine.ToString() } else { "N/A" })
    $resultObj | Add-Member -MemberType NoteProperty -Name "Speedup" -Value $(if ($speedupLine) { $speedupLine.ToString() } else { "N/A" })
 
    $results += $resultObj
 
    $output | Out-File "$runFolder\case_$($c.n)_output.txt"
 
    Write-Host "  -> $status"
    Write-Host ""
}
 
$results | Format-Table -AutoSize
$results | Export-Csv "$runFolder\sweep_results.csv" -NoTypeInformation
 
Write-Host ""
Write-Host "Done. Results in $runFolder\sweep_results.csv, per-case logs in $runFolder\"