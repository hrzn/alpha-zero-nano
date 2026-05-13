// Browser-side port of connect4/connect4.py.
//
// Same duck-typed interface as the Python class. Board is a flat
// Int8Array of length 42 in row-major order: index = r * 7 + c.
// Player is +1 or -1.

export class Connect4 {
  readonly rowCount = 6;
  readonly columnCount = 7;
  readonly actionSize = 7;
  readonly numChannels = 3;

  getInitialState(): Int8Array {
    return new Int8Array(this.rowCount * this.columnCount);
  }

  copy(state: Int8Array): Int8Array {
    return new Int8Array(state);
  }

  private at(state: Int8Array, r: number, c: number): number {
    return state[r * this.columnCount + c];
  }

  private setCell(state: Int8Array, r: number, c: number, v: number): void {
    state[r * this.columnCount + c] = v;
  }

  /** Drop a piece in column `action`. Mutates and returns state. */
  updateState(state: Int8Array, action: number, player: number): Int8Array {
    for (let r = this.rowCount - 1; r >= 0; r--) {
      if (this.at(state, r, action) === 0) {
        this.setCell(state, r, action, player);
        return state;
      }
    }
    throw new Error(`Column ${action} is full`);
  }

  getValidMoves(state: Int8Array): Uint8Array {
    const valid = new Uint8Array(this.actionSize);
    for (let c = 0; c < this.columnCount; c++) {
      valid[c] = this.at(state, 0, c) === 0 ? 1 : 0;
    }
    return valid;
  }

  /** Did the piece most recently dropped in `action` complete a 4-in-a-row? */
  checkWin(state: Int8Array, action: number): boolean {
    const col = action;
    let row = -1;
    for (let r = 0; r < this.rowCount; r++) {
      if (this.at(state, r, col) !== 0) { row = r; break; }
    }
    if (row === -1) return false;
    const player = this.at(state, row, col);
    const dirs: [number, number][] = [[0, 1], [1, 0], [1, 1], [1, -1]];
    for (const [dr, dc] of dirs) {
      let count = 1;
      let r = row + dr, c = col + dc;
      while (r >= 0 && r < this.rowCount && c >= 0 && c < this.columnCount && this.at(state, r, c) === player) {
        count++; r += dr; c += dc;
      }
      r = row - dr; c = col - dc;
      while (r >= 0 && r < this.rowCount && c >= 0 && c < this.columnCount && this.at(state, r, c) === player) {
        count++; r -= dr; c -= dc;
      }
      if (count >= 4) return true;
    }
    return false;
  }

  getValueAndTerminated(state: Int8Array, action: number): { value: number; terminated: boolean } {
    if (this.checkWin(state, action)) return { value: 1, terminated: true };
    const valid = this.getValidMoves(state);
    let any = 0;
    for (let i = 0; i < valid.length; i++) any |= valid[i];
    if (!any) return { value: 0, terminated: true };
    return { value: 0, terminated: false };
  }

  getOpponent(player: number): number {
    return -player;
  }

  /** Encode as 3 channels (own, opponent, empty), shape (3, 6, 7), C-order flat. */
  encodeState(state: Int8Array, player: number): Float32Array {
    const enc = new Float32Array(this.numChannels * this.rowCount * this.columnCount);
    const planeSize = this.rowCount * this.columnCount;
    for (let r = 0; r < this.rowCount; r++) {
      for (let c = 0; c < this.columnCount; c++) {
        const v = this.at(state, r, c);
        const idx = r * this.columnCount + c;
        if (v === player) enc[idx] = 1;
        else if (v === -player) enc[planeSize + idx] = 1;
        else enc[2 * planeSize + idx] = 1;
      }
    }
    return enc;
  }
}
