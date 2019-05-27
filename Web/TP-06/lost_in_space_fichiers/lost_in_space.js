"use strict";

let game;

class Game {
	constructor( width, height ) {
		this.width = width;
		this.height = height;
	}

	initPlayerShip() {
		this.playerShip = new Ship( this.width/2, this.height );
	}
}

class Ship {
	constructor( x, y ) {
		this.x = x;
		this.y = y;
		this.speedX = 0;
		this.speedY = 0;
		this.accelerationX = 0;
		this.accelerationY = 0;
	}

	drawShip( ctx ) {
		ctx.strokeStyle = "#FF0000"; 
		ctx.moveTo(this.x, this.y);
		ctx.lineTo(this.x+20, this.y+50);
		ctx.lineTo(this.x-20, this.y+50);
		ctx.lineTo(this.x, this.y);
		ctx.stroke();
	}
}

function playerMoveShip( event ) {
	if ( event.key === "ArrowUp" ) {
		game.playerShip.y -= 1;
	}
	else if ( event.key === "ArrowDown" ) {
		game.playerShip.y += 1;
	}
	else if ( event.key === "ArrowLeft" ) {
		game.playerShip.x -= 1;
	}
	else if ( event.key === "ArrowRight" ) {
		game.playerShip.x += 1;
	}
}

window.onload = function() {
	let canvas = document.getElementById("game_area");
	let context = canvas.getContext("2d");
	game = new Game( 600, 600 );
	game.initPlayerShip();
	window.addEventListener("keydown", playerMoveShip );

	game.playerShip.drawShip( context );

	this.interval = setInterval(() => {
		game.playerShip.drawShip( context );
	}, 20);
};
