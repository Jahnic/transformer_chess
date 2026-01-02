"""
Chess data preprocessing for transformer training.

Converts PGN files into clean move sequences suitable for character-level
language modeling.
"""

import re
import os
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass
from collections import Counter
import json
import random


@dataclass
class ChessGame:
    """Represents a single chess game."""
    moves: str
    result: str
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    

def parse_pgn_file(filepath: str) -> Iterator[ChessGame]:
    """
    Parse a PGN file and yield individual games.
    
    Handles the standard PGN format with headers in brackets
    followed by move text.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split into individual games (each game starts with [Event)
    games = re.split(r'\n(?=\[Event )', content)
    
    for game_text in games:
        if not game_text.strip():
            continue
            
        # Extract headers
        headers = {}
        for match in re.finditer(r'\[(\w+)\s+"([^"]+)"\]', game_text):
            headers[match.group(1)] = match.group(2)
        
        # Extract moves (everything after the headers)
        # Remove headers and comments
        moves_section = re.sub(r'\[.*?\]', '', game_text)
        
        # Clean up the moves
        moves = clean_moves(moves_section)
        
        if moves:  # Only yield if we have actual moves
            yield ChessGame(
                moves=moves,
                result=headers.get('Result', '*'),
                white_elo=safe_int(headers.get('WhiteElo')),
                black_elo=safe_int(headers.get('BlackElo'))
            )


def safe_int(value: Optional[str]) -> Optional[int]:
    """Safely convert string to int."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def clean_moves(moves_text: str) -> str:
    """
    Clean move text into a simple sequence.
    
    Input:  "1. e4 e5 2. Nf3 Nc6 3. Bb5 {comment} a6 1-0"
    Output: "e4 e5 Nf3 Nc6 Bb5 a6"
    """
    # Remove comments in curly braces
    moves = re.sub(r'\{[^}]*\}', '', moves_text)
    
    # Remove comments after semicolons (to end of line)
    moves = re.sub(r';.*$', '', moves, flags=re.MULTILINE)
    
    # Remove move numbers (1. or 1... or 12. etc)
    moves = re.sub(r'\d+\.+\s*', '', moves)
    
    # Remove result markers
    moves = re.sub(r'1-0|0-1|1/2-1/2|\*', '', moves)
    
    # Remove NAG symbols ($1, $2, etc - annotation symbols)
    moves = re.sub(r'\$\d+', '', moves)
    
    # Remove variation markers (parentheses and their content)
    # This is simplified - nested variations would need recursive handling
    moves = re.sub(r'\([^)]*\)', '', moves)
    
    # Normalize whitespace
    moves = ' '.join(moves.split())
    
    return moves.strip()


def filter_by_elo(games: Iterator[ChessGame], min_elo: int = 1800) -> Iterator[ChessGame]:
    """Filter games to only include those with both players above min_elo."""
    for game in games:
        if game.white_elo and game.black_elo:
            if game.white_elo >= min_elo and game.black_elo >= min_elo:
                yield game
        else:
            # If no ELO info, include the game (might be from titled players)
            yield game


def format_for_training(game: ChessGame, include_result: bool = False) -> str:
    """
    Format a game for training.
    
    Options:
    - Just moves: "e4 e5 Nf3 Nc6"
    - With result: "e4 e5 Nf3 Nc6 <1-0>"
    """
    if include_result and game.result in ['1-0', '0-1', '1/2-1/2']:
        return f"{game.moves} <{game.result}>"
    return game.moves


def build_vocabulary(text: str) -> dict:
    """
    Build character-level vocabulary from text.
    
    Returns mapping of character -> index.
    """
    chars = sorted(set(text))
    
    # Reserve special tokens
    vocab = {
        '<pad>': 0,
        '<sos>': 1,  # Start of sequence
        '<eos>': 2,  # End of sequence
    }
    
    # Add all characters
    for i, char in enumerate(chars):
        vocab[char] = i + 3
    
    return vocab


def analyze_vocabulary(text: str) -> dict:
    """
    Analyze the vocabulary and return statistics.
    """
    char_counts = Counter(text)
    total_chars = len(text)
    
    return {
        'unique_chars': len(char_counts),
        'total_chars': total_chars,
        'char_frequencies': {
            char: count / total_chars 
            for char, count in char_counts.most_common()
        },
        'most_common': char_counts.most_common(20),
        'sample': text[:500]
    }


def process_data(
    input_path: str,
    output_dir: str,
    min_elo: int = 1800,
    train_ratio: float = 0.9,
    max_games: Optional[int] = None,
    include_result: bool = False
) -> dict:
    """
    Full preprocessing pipeline.
    
    Args:
        input_path: Path to PGN file
        output_dir: Directory for output files
        min_elo: Minimum ELO for filtering
        train_ratio: Fraction of data for training
        max_games: Maximum number of games to process
        include_result: Whether to append game result
    
    Returns:
        Statistics dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse and filter games
    print(f"Parsing {input_path}...")
    games = list(parse_pgn_file(input_path))
    print(f"Found {len(games)} games")
    
    # Filter by ELO
    games = list(filter_by_elo(iter(games), min_elo))
    print(f"After ELO filter (>= {min_elo}): {len(games)} games")
    
    # Limit if specified
    if max_games and len(games) > max_games:
        random.shuffle(games)
        games = games[:max_games]
        print(f"Limited to {max_games} games")
    
    # Format for training
    formatted = [format_for_training(g, include_result) for g in games]
    
    # Filter out empty games and very short games
    formatted = [g for g in formatted if len(g.split()) >= 6]  # At least 3 moves each
    print(f"After filtering short games: {len(formatted)} games")
    
    # Shuffle
    random.shuffle(formatted)
    
    # Split train/val
    split_idx = int(len(formatted) * train_ratio)
    train_games = formatted[:split_idx]
    val_games = formatted[split_idx:]
    
    # Join with newlines
    train_text = '\n'.join(train_games)
    val_text = '\n'.join(val_games)
    
    # Build vocabulary from training data
    vocab = build_vocabulary(train_text)
    
    # Analyze
    stats = analyze_vocabulary(train_text)
    stats['num_train_games'] = len(train_games)
    stats['num_val_games'] = len(val_games)
    stats['train_chars'] = len(train_text)
    stats['val_chars'] = len(val_text)
    
    # Save outputs
    train_path = output_dir / 'train.txt'
    val_path = output_dir / 'val.txt'
    vocab_path = output_dir / 'vocab.json'
    stats_path = output_dir / 'stats.json'
    
    with open(train_path, 'w') as f:
        f.write(train_text)
    print(f"Saved training data to {train_path} ({len(train_text):,} chars)")
    
    with open(val_path, 'w') as f:
        f.write(val_text)
    print(f"Saved validation data to {val_path} ({len(val_text):,} chars)")
    
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to {vocab_path} ({len(vocab)} tokens)")
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved statistics to {stats_path}")
    
    return stats


# For quick testing with synthetic data
def create_sample_data(output_dir: str, num_games: int = 1000) -> None:
    """
    Create synthetic chess-like data for testing the pipeline.
    Uses common opening moves to create plausible-looking games.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Common opening moves and responses
    white_first = ['e4', 'd4', 'Nf3', 'c4', 'g3']
    black_responses = {
        'e4': ['e5', 'c5', 'e6', 'c6', 'd5', 'Nf6'],
        'd4': ['d5', 'Nf6', 'e6', 'f5'],
        'Nf3': ['d5', 'Nf6', 'c5'],
        'c4': ['e5', 'Nf6', 'c5', 'e6'],
        'g3': ['d5', 'Nf6', 'e5']
    }
    
    # Common continuations
    common_moves = [
        'Nc3', 'Bc4', 'Bb5', 'Be2', 'Bd3', 'Nf3', 'Bg5', 'O-O', 'O-O-O',
        'Re1', 'Qe2', 'Qd3', 'd3', 'c3', 'a3', 'h3', 'b3',
        'Nc6', 'Bc5', 'Be7', 'Bd6', 'Nf6', 'Bg4', 'O-O',
        'Re8', 'Qe7', 'd6', 'c6', 'a6', 'h6', 'b6',
        'Bxf6', 'Nxd5', 'exd5', 'Bxc6', 'Nxe5', 'dxe5',
        'Kh1', 'Kh8', 'Rfe1', 'Rfe8', 'Rad1', 'Rad8',
    ]
    
    games = []
    for _ in range(num_games):
        moves = []
        
        # Opening
        first_move = random.choice(white_first)
        moves.append(first_move)
        response = random.choice(black_responses.get(first_move, ['Nf6']))
        moves.append(response)
        
        # Continue with random common moves
        num_additional = random.randint(10, 40)
        moves.extend(random.choices(common_moves, k=num_additional))
        
        games.append(' '.join(moves))
    
    # Split
    random.shuffle(games)
    split_idx = int(len(games) * 0.9)
    train_games = games[:split_idx]
    val_games = games[split_idx:]
    
    train_text = '\n'.join(train_games)
    val_text = '\n'.join(val_games)
    
    # Build vocab and save
    vocab = build_vocabulary(train_text)
    stats = analyze_vocabulary(train_text)
    stats['num_train_games'] = len(train_games)
    stats['num_val_games'] = len(val_games)
    stats['synthetic'] = True
    
    with open(output_dir / 'train.txt', 'w') as f:
        f.write(train_text)
    with open(output_dir / 'val.txt', 'w') as f:
        f.write(val_text)
    with open(output_dir / 'vocab.json', 'w') as f:
        json.dump(vocab, f, indent=2)
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Created synthetic dataset with {num_games} games")
    print(f"Training: {len(train_text):,} chars, Validation: {len(val_text):,} chars")
    print(f"Vocabulary size: {len(vocab)} tokens")


if __name__ == '__main__':
    import sys
    
    # Default paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'
    
    # Check if we have real data
    pgn_files = list(raw_dir.glob('*.pgn'))
    
    if pgn_files:
        # Process real data
        print("Found PGN files, processing real data...")
        input_file = pgn_files[0]  # Use first PGN file found
        stats = process_data(
            input_path=str(input_file),
            output_dir=str(processed_dir),
            min_elo=1800,
            max_games=50000
        )
    else:
        # Create synthetic data for testing
        print("No PGN files found, creating synthetic data for testing...")
        create_sample_data(str(processed_dir), num_games=5000)
    
    print("\n=== Data Preprocessing Complete ===")
    print(f"Output directory: {processed_dir}")
    print("\nFiles created:")
    for f in processed_dir.glob('*'):
        print(f"  - {f.name}")
