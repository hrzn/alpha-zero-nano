"""Chess game environment wrapping python-chess.

Action encoding: from_square * 64 + to_square (4096 actions total).
Pawn promotions default to queen. Underpromotions are not supported.

Player convention: White = 1, Black = -1.
"""

import chess
import numpy as np


class ChessGame:
    # Piece type order used in encoding (index within player's 6 channels)
    _PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                    chess.ROOK, chess.QUEEN, chess.KING]

    def __init__(self):
        self.row_count = 8
        self.column_count = 8
        self.action_size = 64 * 64
        self.num_channels = 17  # 6 own + 6 opp + 4 castling + 1 colour

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_initial_state(self) -> chess.Board:
        return chess.Board()

    def update_state(self, state: chess.Board, action: int, player) -> chess.Board:
        """Return a new board with the action applied. Does not mutate state."""
        new_state = state.copy()
        move = self._action_to_move(new_state, action)
        new_state.push(move)
        return new_state

    def get_valid_moves(self, state: chess.Board) -> np.ndarray:
        """Return a binary mask of shape (action_size,) for legal moves.

        Actions are encoded in the current player's spatial frame (flipped
        for black) to match encode_state.
        """
        mask = np.zeros(self.action_size, dtype=np.uint8)
        flip = (state.turn == chess.BLACK)
        for move in state.legal_moves:
            mask[self._move_to_action(move, flip=flip)] = 1
        return mask

    def get_value_and_terminated(self, state: chess.Board, action) -> tuple[float, bool]:
        """Return (value, terminated).

        value = 1 if the side that just moved won (checkmate),
                0 for draws / ongoing.
        """
        if state.is_checkmate():
            return 1.0, True
        if (state.is_stalemate()
                or state.is_insufficient_material()
                or state.is_seventyfive_moves()
                or state.is_fivefold_repetition()):
            return 0.0, True
        return 0.0, False

    def check_win(self, state: chess.Board, action) -> bool:
        return state.is_checkmate()

    def get_opponent(self, player: int) -> int:
        return -player

    def state_hash(self, state: chess.Board) -> int:
        return hash(state.fen())

    # ------------------------------------------------------------------
    # Action encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flip_sq(sq: int) -> int:
        """Flip rank: a1↔a8, e2↔e7, etc.  Used so that black's actions
        are in the same spatially-flipped frame as encode_state."""
        return sq ^ 56

    def _move_to_action(self, move: chess.Move, flip: bool = False) -> int:
        from_sq = move.from_square
        to_sq = move.to_square
        if flip:
            from_sq = self._flip_sq(from_sq)
            to_sq = self._flip_sq(to_sq)
        return from_sq * 64 + to_sq

    def _action_to_move(self, state: chess.Board, action: int) -> chess.Move:
        from_sq = action // 64
        to_sq = action % 64
        if state.turn == chess.BLACK:
            from_sq = self._flip_sq(from_sq)
            to_sq = self._flip_sq(to_sq)
        move = chess.Move(from_sq, to_sq)
        # Promote to queen if this is a pawn reaching the back rank
        piece = state.piece_at(from_sq)
        if (piece is not None
                and piece.piece_type == chess.PAWN
                and chess.square_rank(to_sq) in (0, 7)):
            move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
        return move

    def encode_state(self, state: chess.Board, player: int) -> np.ndarray:
        """Encode board as 17 channels from the given player's perspective.

        Channels 0–5:   current player's pieces (P, N, B, R, Q, K)
        Channels 6–11:  opponent's pieces
        Channels 12–15: castling rights (own K-side, own Q-side, opp K-side, opp Q-side)
        Channel 16:     colour plane (1.0 if white to move, 0.0 if black)

        The board is always oriented from the current player's perspective:
        rank 1 is at row index 0 for white, rank 8 is at row index 0 for black.
        """
        white_to_move = (player == 1)
        own_colour  = chess.WHITE if white_to_move else chess.BLACK
        opp_colour  = chess.BLACK if white_to_move else chess.WHITE

        encoded = np.zeros((17, 8, 8), dtype=np.float32)

        for i, piece_type in enumerate(self._PIECE_TYPES):
            for sq in state.pieces(piece_type, own_colour):
                r, c = chess.square_rank(sq), chess.square_file(sq)
                if not white_to_move:
                    r = 7 - r  # flip rank for black's perspective
                encoded[i, r, c] = 1.0

            for sq in state.pieces(piece_type, opp_colour):
                r, c = chess.square_rank(sq), chess.square_file(sq)
                if not white_to_move:
                    r = 7 - r
                encoded[6 + i, r, c] = 1.0

        # Castling rights (broadcast as full planes)
        if white_to_move:
            own_ks  = state.has_kingside_castling_rights(chess.WHITE)
            own_qs  = state.has_queenside_castling_rights(chess.WHITE)
            opp_ks  = state.has_kingside_castling_rights(chess.BLACK)
            opp_qs  = state.has_queenside_castling_rights(chess.BLACK)
        else:
            own_ks  = state.has_kingside_castling_rights(chess.BLACK)
            own_qs  = state.has_queenside_castling_rights(chess.BLACK)
            opp_ks  = state.has_kingside_castling_rights(chess.WHITE)
            opp_qs  = state.has_queenside_castling_rights(chess.WHITE)

        encoded[12] = float(own_ks)
        encoded[13] = float(own_qs)
        encoded[14] = float(opp_ks)
        encoded[15] = float(opp_qs)

        # Colour plane
        encoded[16] = 1.0 if white_to_move else 0.0

        return encoded

    def uci_to_action(self, uci: str, player: int = 1) -> int:
        """Convert a UCI string (e.g. 'e2e4') to an action integer.

        player: 1 (white) or -1 (black).  Actions are flipped for black
        to match the encoding frame.
        """
        move = chess.Move.from_uci(uci)
        return self._move_to_action(move, flip=(player == -1))

    def action_to_uci(self, action: int, player: int = 1) -> str:
        """Convert an action integer to a UCI string.

        player: 1 (white) or -1 (black).  Un-flips for black.
        """
        from_sq = action // 64
        to_sq = action % 64
        if player == -1:
            from_sq = self._flip_sq(from_sq)
            to_sq = self._flip_sq(to_sq)
        return chess.Move(from_sq, to_sq).uci()
