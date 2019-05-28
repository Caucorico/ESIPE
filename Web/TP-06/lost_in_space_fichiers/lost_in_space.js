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
		ctx.beginPath();
		ctx.moveTo(this.x, this.y);
		ctx.lineTo(this.x+20, this.y+50);
		ctx.lineTo(this.x-20, this.y+50);
		ctx.lineTo(this.x, this.y);
		ctx.stroke();
	}

	applyAcceleration() {
		this.speedX = clamp(this.speedX + this.accelerationX, -5, 5);
		this.speedY = clamp(this.speedY + this.accelerationY, -5, 5);
	}

	applySpeed() {
		this.x += this.speedX;
		this.y += this.speedY;
	}
}

function clamp( a, min, max )
{
	if ( a > max ) return max;
	else if ( a < min ) return min;
	else return a;
}

function playerMoveShip( event ) {
	if ( event.key === "ArrowUp" ) {
		game.playerShip.accelerationY += game.playerShip.accelerationY-0.5;
	}
	else if ( event.key === "ArrowDown" ) {
		game.playerShip.accelerationY = game.playerShip.accelerationY+0.5;
	}
	else if ( event.key === "ArrowLeft" ) {
		game.playerShip.accelerationX = game.playerShip.accelerationX-0.5;
	}
	else if ( event.key === "ArrowRight" ) {
		game.playerShip.accelerationX = game.playerShip.accelerationX+0.5;
	}
}

function playerStopMoveShip( event )
{
	if ( event.key === "ArrowUp" ) {
		game.playerShip.accelerationY = 0;
	}
	else if ( event.key === "ArrowDown" ) {
		game.playerShip.accelerationY = 0;
	}
	else if ( event.key === "ArrowLeft" ) {
		game.playerShip.accelerationX = 0;
	}
	else if ( event.key === "ArrowRight" ) {
		game.playerShip.accelerationX = 0;
	}
}

window.onload = function() {
	let canvas = document.getElementById("game_area");
	let context = canvas.getContext("2d");
	context.fillStyle = "black";
	game = new Game( 600, 600 );
	game.initPlayerShip();
	/*window.addEventListener("keydown", playerMoveShip );
	window.addEventListener("keyup", playerStopMoveShip );*/

	let map = {}; // You could also use an array
	onkeydown = onkeyup = function(e){
	    e = e || event; // to deal with IE
	    map[e.keyCode] = e.type == 'keydown';
	    console.log(map);

	    if ( map[38] && map[40] ) /* up + down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( !map[38] && !map[40] ) /* ~up + ~down*/
	    {
	    	game.playerShip.accelerationY = 0;
	    }

	    if ( map[37] && map[39] ) /* left + right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }

	    if ( !map[37] && !map[39] ) /* ~left + ~right*/
	    {
	    	game.playerShip.accelerationX = 0;
	    }
	    
	    if ( map[38] ) {/* up */
	    	game.playerShip.accelerationY = game.playerShip.accelerationY-0.5;
	    }
	    if ( map[40] ) {/* down */
	    	game.playerShip.accelerationY = game.playerShip.accelerationY+0.5;
	    }

	    if ( map[37] ) {/* left */
	    	game.playerShip.accelerationX = game.playerShip.accelerationX-0.5;
	    }
	    if ( map[39] ) {/* right */
	    	game.playerShip.accelerationX = game.playerShip.accelerationX+0.5;
	    }
	}

	context.save();
	game.playerShip.drawShip( context );

	this.interval = setInterval(() => {
		context.clearRect(0,0,600,600);
		game.playerShip.applyAcceleration();
		game.playerShip.applySpeed();
		game.playerShip.drawShip( context );
	}, 20);
};
