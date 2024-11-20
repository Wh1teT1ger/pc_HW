/nested_hash 3b2086feb2460e20769071f1de5413d56c32eec40a635801df7e57f92fa665b3 SHA256,SHA256,SHA256 --max-length 6
Start
Start chunk 0
Start chunk 1
Start chunk 2
Start chunk 3
Start chunk 4
Match found: zzzzzz
Execution time: 59045 ms

nested_hash 9918da8d1bdf61831d0a8460b920975186cdfcba3e5cc881a4ef0e6a3c406c78 SHA256,SHA256,S
HA256 --max-length 5
Start
Start chunk 0
Match found: zzzzz
Execution time: 2227 ms

nested_hash 76674e59eaac2e79cb77541da2b29bd11457c906a66b452d6a6a08f2a272775d SHA256,SHA256,S
HA256 --max-length 4
Start
Start chunk 0
Match found: zzzz
Execution time: 96 ms

./nested_hash -t 76674e59eaac2e79cb77541da2b29bd11457c906a66b452d6a6a08f2a272775d -s sha256 sha256 sha256 -m 4

./nested_hash -t 9918da8d1bdf61831d0a8460b920975186cdfcba3e5cc881a4ef0e6a3c406c78 -s sha256 sha256 sha256 -m 5