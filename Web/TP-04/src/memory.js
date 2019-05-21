
function MemoryGame(images, blank) {
  this.images = images;
  this.blank = blank;
  this.card1 = null;
  this.card2 = null;
}

MemoryGame.prototype.build = function build(div) {
  /*div.innerHTML = this.images.length;*/
  this.cards = shuffleCards(this.images.length);
  this.turnBacks = {};

  for ( let i = 0 ; i < this.cards.length ; i++ )
  	this.turnBacks[i] = false;

  for ( let i = 0 ; i < this.cards.length ; i++ ) {
  	let card = this.cards[i];
  	let childDiv = document.createElement("div");
  	let image = document.createElement("img");
  	image.setAttribute("src", "../images/" + this.blank);
  	childDiv.appendChild(image);
  	childDiv.setAttribute("data-index", i); 
  	div.appendChild(childDiv);

  	childDiv.setAttribute("onclick", "clicked(this, game)");
  }
}

MemoryGame.prototype.getState = function getState() {
	if ( this.card1 == null && this.card2 == null  ) return 0;
	else if ( this.card1 != null && this.card2 == null) return 1;
	else return 2;
}

function clicked(element, game) {

	console.log(game.getState());

	if ( !game.turnBacks[element.getAttribute("data-index")] ) {
		element.firstChild.src = "../images/" + game.images[game.cards[element.getAttribute("data-index")]];
		game.turnBacks[element.getAttribute("data-index")] = true;

		if ( game.getState() == 0 ) game.card1 = element;
		else if ( game.getState() == 1 ) {
			game.card2 = element;

			if ( game.cards[game.card1.getAttribute("data-index")] == game.cards[game.card2.getAttribute("data-index")] ) {
				game.card1 = null;
				game.card2 = null;
			}
			else {
				setTimeout(() => {
					game.card1.firstChild.src = "../images/" + game.blank;
					game.card2.firstChild.src = "../images/" + game.blank;
					game.turnBacks[game.card1.getAttribute("data-index")] = false;
					game.turnBacks[game.card2.getAttribute("data-index")] = false;
					game.card1 = null;
					game.card2 = null;
				}, 500);
			}
		}

	}
}

function swap( cards, x, y ) {
	let buff;
	buff = cards[x];
	cards[x] = cards[y];
	cards[y] = buff;
}

function shuffleCards(length) {
  let cards = [];
  for(let i = 0; i < length; i++) {
    cards.push(i);
    cards.push(i);
  }

  for( let i = 0 ; i < cards.length ; i++ )
  {
  	swap( cards, Math.floor(Math.random() * cards.length), Math.floor(Math.random() * cards.length) );
  }

  return cards;
}
