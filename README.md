## Dependencies
- Python 3.8.18
- PyTorch 2.4.1+cu121
## Implementation of Bri4CDSR
or the Toy->Game setting:
```
python main.py --device=cuda:0 --dataset=toy-game --target_dataset=game --w=0.3 --diffusion_loss_weight=0.5 --interval=37868 --random_seed=2025
```
or the Mucic->Book setting:
```
python main.py --device=cuda:0 --dataset=douban_music_book --target_dataset=douban_book --w=0.6 --diffusion_loss_weight=0.1 --interval=50448 --random_seed=2025
```
---